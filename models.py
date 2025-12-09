import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import sys
from functools import partial
from torch import einsum
from einops import rearrange
from utils import *
from seq2seq import Encoder, Decoder, Seq2Seq
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
    

class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class MlpBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, int(self.input_dim))

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        return self.fc2(y)


class MixerBlock(nn.Module):
    def __init__(self, hidden_len, hidden_dim, tokens_dim, channels_len):
        super().__init__()
        self.tokens_dim = tokens_dim
        self.hidden_len = hidden_len
        self.channels_len = channels_len 
        self.hidden_dim = hidden_dim

        self.token_mixer = MlpBlock(self.hidden_len, self.tokens_dim)   
        self.channel_mixer = MlpBlock(self.tokens_dim, self.channels_len)

    def forward(self, x):
        y = F.layer_norm(x, x.shape[1:])
        y = torch.transpose(y, 1, 2) 
        y = self.token_mixer(y) 
        y = torch.transpose(y, 1, 2) 
        x = x + y
        y = F.layer_norm(x, x.shape[1:])
        return x + self.channel_mixer(y) 


class MlpMixer(nn.Module):
    def __init__(self, feature_length, hidden_len, hidden_dim, tokens_dim, channels_len, num_blocks=2):
        super().__init__()
        self.feature_length = feature_length
        self.num_blocks = num_blocks 
        self.hidden_len = hidden_len 
        self.hidden_dim = hidden_dim 
        self.tokens_dim = tokens_dim 
        self.channels_len = channels_len 

        
        self.conv = nn.Conv1d(in_channels=self.feature_length,
                              out_channels=self.hidden_len,
                              kernel_size=1)

        for nb in range(self.num_blocks):
            setattr(self, "mixerBlock_{}".format(nb), MixerBlock(hidden_len = self.hidden_len, 
                                                                 hidden_dim = self.hidden_dim,
                                                                 tokens_dim = self.tokens_dim, 
                                                                 channels_len = self.channels_len))

        self.fc = nn.Linear(in_features=self.tokens_dim,
                            out_features=self.tokens_dim) # different initialization

    
    def forward(self, inputs):
        x = self.conv(inputs)
        for nb in range(self.num_blocks):
            x = getattr(self, "mixerBlock_{}".format(nb))(x) # b * hidden_len * dim

        x = F.layer_norm(x, x.shape[1:])

        # x = torch.transpose(x, 1, 2)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.fc(x) # b * dim

        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CEBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp2 = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.mlp1(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp2(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.conv1d = nn.Conv1d(in_channels=4,
                                out_channels=out_channels,
                                kernel_size=1)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # pdb.set_trace()
        x = self.conv1d(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CEDenoiser(nn.Module):
    def __init__(
        self,
        out_channels=4,
        hidden_dim=400,
        depth=2,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([
            CEBlock(hidden_dim=hidden_dim, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_dim, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, c):
        t = self.t_embedder(t)                   # (N, D)
        c = torch.mean(c, dim=1)                 # (N, D)
        c = t + c                                # (N, D)
        for block in self.blocks:
            x = block(x, c)
        # pdb.set_trace()                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, 1, D)
        x = x.squeeze(1)

        return x


class CEDenoise_Net(nn.Module):  
    def __init__(self,
                 input_dim=200,
                 with_time_emb=True,
                 max_seq_len=20,
                 modelconfigs=None
    ):
        super().__init__()
        # pdb.set_trace()
        assert modelconfigs is not None
        self.model_name = modelconfigs['Denoise_Net']['name']
        self.modelconfigs = modelconfigs['Denoise_Net']
        # time embedding
        dim = 64
        time_dim = input_dim
        if with_time_emb:
            self.time_encoder = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        self.dense_fn = nn.Linear(max_seq_len*input_dim, max_seq_len*input_dim)
        self.dense_fn2 = nn.Linear(input_dim*max_seq_len, input_dim)

        if 'with_position_emb' in self.modelconfigs.keys():
            self.register_buffer("position_ids", torch.arange(
                max_seq_len).expand((1, -1)))
            self.position_embeddings = nn.Embedding(
                max_seq_len, input_dim)

            # self-attension
            if self.model_name == 'transformer':
                te_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                      nhead=self.modelconfigs['nhead'])
                self.encoder = nn.TransformerEncoder(encoder_layer=te_layer,
                                                     num_layers=self.modelconfigs['blocks_num'])
            elif self.model_name == 'lstm':
                self.encoder = nn.LSTM(input_size=input_dim,
                                       hidden_size=input_dim,
                                       batch_first=False,
                                       num_layers=2)
            
            
            elif self.model_name == 'CEDenoiser':
                self.CEDenoiser = CEDenoiser(out_channels=1,
                               hidden_dim=400,
                               depth=2,
                               mlp_ratio=4.0,)

            elif self.model_name == 'mlp-mixer':
                self.mlpmixer = MlpMixer(feature_length=max_seq_len,  
                                         hidden_len=max_seq_len*100,
                                         hidden_dim=input_dim, 
                                         tokens_dim=input_dim, 
                                         channels_len=max_seq_len*100)

            elif self.model_name == 'seq2seq':
                self.seq2seq_encoder = Encoder(input_size=input_dim,
                                               hidden_size=256)
                self.seq2seq_decoder = Decoder(output_size=input_dim,
                                               hidden_size=256)
                self.encoder = Seq2Seq(self.seq2seq_encoder,
                                       self.seq2seq_decoder)

            
        else:
            self.input_drop = nn.Dropout(self.modelconfigs['input_drop'])
            self.hidden_drop = nn.Dropout(self.modelconfigs['hidden_drop'])
            self.feature_map_drop = nn.Dropout(self.modelconfigs['feat_drop'])
            self.emb_dim1 = self.modelconfigs['embedding_dim1']
            self.emb_dim2 = input_dim // self.emb_dim1
            self.conv1 = nn.Conv2d(in_channels=5, 
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=0,
                                   bias=self.modelconfigs['use_bias'])
            self.bn0 = torch.nn.BatchNorm2d(5)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.dense_layer = nn.Linear(self.modelconfigs['hidden_size'], input_dim * 3) 


    def forward(self, 
                noisy_input_t, 
                condition,
                time,
                mode):
    
        c1_emb, c2_emb, c3_emb = condition
        if mode == 'head-batch':
            sequence = torch.cat([noisy_input_t, c1_emb, c2_emb, c3_emb], dim=1)
            relation_emb, tail_emb, coarse_head_emb = c1_emb, c2_emb, c3_emb
            
        if mode == 'tail-batch':
            sequence = torch.cat([c1_emb, c2_emb, noisy_input_t, c3_emb], dim=1)
            head_emb, relation_emb, coarse_tail_emb = c1_emb, c2_emb, c3_emb


        seq_length = sequence.shape[1]
        # position embedding
        if self.model_name == 'transformer':
            position_ids = self.position_ids[:, :seq_length]
            time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) 

            sequence = self.position_embeddings(position_ids) + sequence + time_embeddings
            sequence = sequence.permute(1, 0, 2)                       
            sequence = self.encoder(sequence)
            sequence = sequence.permute(1, 0, 2)  
            sequence = sequence.reshape(sequence.size(0), -1)
            return denoise_y0 

        elif self.model_name == 'mlp':
            sequence = self.dense_fn(sequence)  
            sequence = sequence.reshape(sequence.size(0), -1)
            denoise_y0 = self.dense_fn2(sequence)
            return denoise_y0

        elif self.model_name == 'CEDenoiser':
            noisy_input_t = noisy_input_t.unsqueeze(1)
            c1_emb, c2_emb, c3_emb = c1_emb.unsqueeze(1), c2_emb.unsqueeze(1), c3_emb.unsqueeze(1)
            if mode == 'head-batch':
                sequence = torch.cat([noisy_input_t, c1_emb, c2_emb, c3_emb], dim=1)
                condition = torch.cat([c1_emb, c2_emb, c3_emb], dim=1)
            if mode == 'tail-batch':
                sequence = torch.cat([c1_emb, c2_emb, noisy_input_t, c3_emb], dim=1)
                condition = torch.cat([c1_emb, c2_emb, c3_emb], dim=1)            
            denoise_y0 = self.CEDenoiser(x=sequence, t=time, c=condition)
            return denoise_y0 



        elif self.model_name == 'mlp-mixer':
            noisy_input_t = noisy_input_t.unsqueeze(1)
            c1_emb, c2_emb, c3_emb = c1_emb.unsqueeze(1), c2_emb.unsqueeze(1), c3_emb.unsqueeze(1)
            if mode == 'head-batch':
                sequence = torch.cat([noisy_input_t, c1_emb, c2_emb, c3_emb], dim=1)
            if mode == 'tail-batch':
                sequence = torch.cat([c1_emb, c2_emb, noisy_input_t, c3_emb], dim=1) 
            denoise_y0 = self.mlpmixer(sequence)  
            return denoise_y0


        elif self.model_name == 'seq2seq':
            time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1)
            sequence = sequence + time_embeddings
            sequence = sequence.permute(1, 0, 2) 
            sequence = self.encoder(sequence)
            sequence = sequence.permute(1, 0, 2)
            return sequence 

        elif self.model_name == 'lstm':
            time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1)
            sequence = sequence + time_embeddings 
            sequence = sequence.permute(1, 0, 2)
            sequence, _ = self.encoder(sequence)
            sequence = sequence.permute(1, 0, 2) 
            sequence = sequence.reshape(sequence.size(0), -1)
            denoise_y0 = self.dense_fn2(sequence)       
            return denoise_y0 

        elif self.model_name == 'conv':
            noisy_input_t = noisy_input_t.view(-1, 3, self.emb_dim1, self.emb_dim2)
            c1_emb, c2_emb = condition
            c1_emb = c1_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)  
            c2_emb = c2_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
            # pdb.set_trace()
            stacked_inputs = torch.cat([noisy_input_t, c1_emb, c2_emb], dim=1) 
            stacked_inputs = self.bn0(stacked_inputs) 
            x = self.input_drop(stacked_inputs)
            x = self.conv1(x) 
            x = self.bn1(x) 
            x = F.leaky_relu(x)
            x = self.feature_map_drop(x)
            x = x.view(x.shape[0], -1) 
            x = self.dense_layer(x) 
            x = x.view(x.shape[0], 3, -1)
            return x


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


if  __name__ == '__main__':
    pass
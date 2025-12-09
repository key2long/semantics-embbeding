import torch
import numpy as np
from functools import partial
from torch import optim
from torch.utils.data import DataLoader
from datasets import *
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import logging
from utils import *
import pdb
from models import *
from multiprocessing import Pool


class KGDM(nn.Module):
    def __init__(self, 
            dataset,
            ce_denoiser,
            nentity,
            nrelation,
            hidden_dim,
            gamma,
            pretrain_emb,
            objective='pred_xtart',
            double_entity_embedding=False,
            double_relation_embedding=False,
            timesteps=200,
            beta_schedule="linear",
            linear_start=1e-4,
            linear_end=2e-2,
            max_seq_len=3,
            loss_type="l2",
            lr=1e-5,
            ddim_sampling_eta=1.,
            use_ensemble = False,
            ddim_sampling_timesteps=10,
            dataset_onehot=False,
            dataset_neg=False,
        ):
        super().__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.pretrain_emb = pretrain_emb
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        if self.pretrain_emb:
            
            if dataset == 'wn18rr':
                self.entity_embedding = nn.Parameter(torch.from_numpy(np.load()).float(), requires_grad=True)
                
                self.relation_embedding = nn.Parameter(torch.from_numpy(np.load()).float(), requires_grad=True)
            
            if dataset == 'FB15K-237':
                self.entity_embedding = nn.Parameter(torch.from_numpy(np.load()).float(), requires_grad=True)
                
                self.relation_embedding = nn.Parameter(torch.from_numpy(np.load()).float(), requires_grad=True)
        else: 
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        self.ce_denoiser = ce_denoiser.cuda()
        # pdb.set_trace()
        self.timesteps = timesteps
        self.objective = objective
        self.ddim_sampling_eta = ddim_sampling_eta
        self.register_schedule(beta_schedule=beta_schedule, linear_start=linear_start, linear_end=linear_end)
        self.loss_type = loss_type
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.use_ensemble = use_ensemble
        self.sampling_timesteps = ddim_sampling_timesteps
        self.dataset_onehot = dataset_onehot
        self.dataset_neg = dataset_neg


    def register_schedule(self, beta_schedule, linear_start, linear_end):

        if beta_schedule == "linear":
            betas = linear_beta_schedule(linear_start, linear_end, timesteps=self.timesteps)
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(1000)
        else:
            raise NotImplementedError("Not supported beta_schedule.")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.register_buffer('posterior_variance', posterior_variance)

    def similarity(self, emb_list):
        for i in range(len(emb_list)-1):
            j = i + 1
            pdb.set_trace()
            cos_sim_i_j = F.cosine_similarity(emb_list[i], emb_list[j], dim=-1)
            print("cos_sim_i_j", cos_sim_i_j)
            l2 = ((emb_list[i] - emb_list[j])**2).sum()
            print("square_sim", l2)


    def forward(self, sample, mode, if_train=False):

        if mode == 'head-batch':
            positive_sample, negative_sample, subsampling_weight, mode = sample
            negative_sample_size = negative_sample.size(1)
            embedding_dim = self.entity_embedding.size(-1)
            batch_size= positive_sample.size(0)
            head = torch.index_select(self.entity_embedding, 
                                      dim=0, 
                                      index=positive_sample[:, 0])
            relation = torch.index_select(self.relation_embedding,
                                          dim=0, index=positive_sample[:, 1])
            tail = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 2])
            t = torch.randint(0, self.timesteps, (batch_size,)).cuda()
            positive_labels_emb = head
            y0_head = tail - relation

            if if_train == True:
                negative_sample_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                loss, positive_sample_loss, negative_sample_loss, metrics = self.cal_losses(ce_denoiser=self.ce_denoiser,
                                       input = (y0_head, positive_labels_emb, negative_sample_emb, subsampling_weight),
                                       condition=(relation, tail, y0_head), 
                                       mode=mode,  
                                       t=t, 
                                       denoise_loss_type=self.loss_type)
                if metrics is not None:
                    return loss, positive_sample_loss, negative_sample_loss, metrics
                else:
                    return loss, positive_sample_loss, negative_sample_loss
                
            if if_train == False:
                negative_sample_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                if self.use_ensemble:
                    pre_embed_list = []
                    score_list = []
                    iter = 20
                    for i in range(iter):
                        pre_embed = self.ddim_sample({'condition':(relation, tail, y0_head), 
                                                      'shape':(batch_size, embedding_dim),
                                                      'mode':mode})
                        pre_embed_list.append(pre_embed)
                        score = torch.norm(negative_sample_emb - pre_embed.unsqueeze(1), p=1, dim=-1)
                        score_list.append(score)

                    mean_pre_embed = sum(pre_embed_list)/len(pre_embed_list)
                    score2 = torch.norm(negative_sample_emb - mean_pre_embed.unsqueeze(1), p=1, dim=-1) # b * 14541 cheak

                    return score_list, score2
            
                else:
                    pre_embed = self.ddim_sample({'condition':(relation, tail, y0_head), 
                                                  'shape':(batch_size, embedding_dim),
                                                  'mode':mode})
                score = torch.norm(negative_sample_emb - pre_embed.unsqueeze(1), p=1, dim=-1)
                score2 = torch.norm(negative_sample_emb - y0_head.unsqueeze(1), p=1, dim=-1)
                denoise_loss = F.smooth_l1_loss(positive_labels_emb, pre_embed)
                return score, score2, denoise_loss.mean(), (relation, tail)


        elif mode == 'tail-batch':
            positive_sample, negative_sample, subsampling_weight, mode = sample
            negative_sample_size = negative_sample.size(1)
            embedding_dim = self.entity_embedding.size(-1)
            batch_size= positive_sample.size(0)
            
            head = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 0])# batch * dim
            relation = torch.index_select(self.relation_embedding,
                                          dim=0, index=positive_sample[:, 1]) # batch * dim
            tail = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 2])# batch * dim
            t = torch.randint(0, self.timesteps, (batch_size,)).cuda()
            positive_labels_emb = tail
            y0_tail = head + relation           
            
            if if_train == True:
                negative_sample_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)

                loss, positive_sample_loss, negative_sample_loss, metrics = self.cal_losses(ce_denoiser=self.ce_denoiser, 
                                       input=(y0_tail, positive_labels_emb, negative_sample_emb, subsampling_weight),
                                       condition=(head, relation, y0_tail), 
                                       mode=mode,
                                       t=t, 
                                       denoise_loss_type=self.loss_type)
                
                if metrics is not None:
                    return loss, positive_sample_loss, negative_sample_loss, metrics
                else:
                    return loss, positive_sample_loss, negative_sample_loss

            else:
                negative_sample_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                if self.use_ensemble:
                    pre_embed_list = []
                    score_list = []
                    iter = 20
                    for i in range(iter):
                        pre_embed = self.ddim_sample({'condition':(head, relation, y0_tail), 
                                                      'shape':(batch_size, embedding_dim),
                                                      'mode':mode})
                        pre_embed_list.append(pre_embed)
                        score = torch.norm(negative_sample_emb - pre_embed.unsqueeze(1), p=1, dim=-1)
                        score_list.append(score)
                    mean_pre_embed = sum(pre_embed_list)/len(pre_embed_list)
                    score2 = torch.norm(negative_sample_emb - mean_pre_embed.unsqueeze(1), p=1, dim=-1) # b * 14541 cheak
                    return score_list, score2
                else:
                    pre_embed = self.ddim_sample({'condition':(head, relation, y0_tail), 
                                                  'shape':(batch_size, embedding_dim),
                                                  'mode':mode})
                score = torch.norm(negative_sample_emb - pre_embed.unsqueeze(1), p=1, dim=-1) 
                score2 = torch.norm(negative_sample_emb - y0_tail.unsqueeze(1), p=1, dim=-1) 
                denoise_loss = F.smooth_l1_loss(positive_labels_emb, pre_embed)
                return score, score2, denoise_loss.mean(), (head, relation)
        else:
            raise ValueError('mode %s not supported' % mode)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )        

    def predict_start_from_noise(self, x_t, t, noise):
        return(
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_sample(self, 
                 input_start, 
                 t, 
                 noise=None):
        if noise is None:
            noise = torch.randn_like(input_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod,
                                        t, 
                                        input_start.shape) 
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                                                  t, 
                                                  input_start.shape)
        noise_input_t = sqrt_alphas_cumprod_t * input_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noise_input_t
    

    def cal_losses(self, 
                   ce_denoiser,
                   input, 
                   condition, 
                   t,
                   mode, 
                   noise=None, 
                   denoise_loss_type="l1"):
        y0_pre, positive_labels_emb, negative_sample_emb, subsampling_weight = input # positive_labels_emb: b*dim; neg: b*neg*dim
        batch_size = positive_labels_emb.size(0)
        negative_sample_size = negative_sample_emb.size(0)
        if noise is None:
            noise = torch.randn_like(y0_pre)
        noisy_input_t = self.q_sample(input_start=y0_pre, 
                                      t=t, 
                                      noise=noise)
        denoise_y0 = ce_denoiser(noisy_input_t, condition, t, mode=mode)
        positive_score = self.gamma - torch.norm(denoise_y0 - positive_labels_emb, p=1, dim=-1)
        negative_score = self.gamma - torch.norm(denoise_y0.unsqueeze(1) - negative_sample_emb, p=1, dim=-1)
        positive_score = F.logsigmoid(positive_score)
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss)/2
        metrics = None      
        return loss, positive_sample_loss, negative_sample_loss, metrics
   
   
    @torch.no_grad()
    def ddim_sample(self,
                    args):
        condition = args['condition']     
        shape = args['shape']
        mode = args['mode']
        batch_size = shape[0]
        total_timesteps, sampling_timesteps, eta, objective = self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps-1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 
        y0_pre = torch.randn(shape).cuda()
        ensemble_triple_embs = []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time).cuda()
            x_start = self.ce_denoiser(y0_pre, condition, time_cond, mode)
            pred_noise = self.predict_noise_from_start(y0_pre, time_cond, x_start)
            if time_next < 0:
                y0_pre = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(y0_pre)
            y0_pre = x_start * alpha_next.sqrt() + \
                         c * pred_noise + \
                         sigma * noise 
        return y0_pre

   
    def p_sample(self, 
                 input_rand, 
                 condition, 
                 t, 
                 t_index, 
                 mode):
        betas_t = extract(self.betas, t, input_rand.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, input_rand.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, input_rand.shape)
    
        pred_start = self.ce_denoiser(input_rand, condition, t, mode)
        
        pred_noise = self.predict_noise_from_start(input_rand, t, pred_start)

        model_mean = sqrt_recip_alphas_t * (input_rand - betas_t * pred_noise 
                                            / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, input_rand.shape)
            noise = torch.randn_like(input_rand)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
  
    def p_sample_loop(self, 
                      condition, 
                      shape, 
                      mode):
        time_steps = self.timesteps
        noise = torch.randn(shape).cuda()
        batch_size = shape[0]
        out = {}
        with torch.no_grad():
            for i in reversed(range(0, time_steps)):
                t = torch.full((batch_size,), i).cuda()
                noise = self.p_sample(input_rand=noise,
                                      condition=condition,
                                      t=t,
                                      t_index=i,
                                      mode=mode)
                out[str(i)] = noise
            recon_embed = out[str(0)]
            return recon_embed


    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator) # pos: batch * 3, neg: batch * neg_num 
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda() 
        loss, positive_sample_loss, negative_sample_loss, = model((positive_sample, negative_sample, subsampling_weight, mode), mode=mode, if_train=True)
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        loss.backward() 
        optimizer.step()


        positive_sample_loss = positive_sample_loss.mean()
        negative_sample_loss = negative_sample_loss.mean()

        log = {
            **regularization_log,
            'loss': loss.item(),
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
        }
        return log


    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, nentity, nrelation):      
        model.eval()
        test_dataloader_head = DataLoader(
                Test_KGDM_Neg_Dataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=Test_KGDM_Neg_Dataset.collate_fn
            )

        test_dataloader_tail = DataLoader(
                Test_KGDM_Neg_Dataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=Test_KGDM_Neg_Dataset.collate_fn
            )
            
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    if model.use_ensemble is not True:
                        score, score2, denoise_loss, condition = model((positive_sample, negative_sample, filter_bias, mode), mode=mode, if_train=False)
                        score = model.gamma.item() - score
                        score2 = model.gamma.item() - score2
                        score += filter_bias
                        score2 += filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=True) # 
                        argsort_direct = torch.argsort(score2, dim = 1, descending=True) # 
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1
                            ranking2 = (argsort_direct[i, :] == positive_arg[i]).nonzero()
                            assert ranking2.size(0) == 1
                            ranking2 = 1 + ranking2.item()
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'Denoise_loss': float(denoise_loss),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })
                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                        step += 1
                    
                    if model.use_ensemble is True:
                        score_list, score2= model((positive_sample, negative_sample, filter_bias, mode), mode=mode, if_train=False) # batch * dim
                        argsort_list = []
                        for score_i in score_list:
                            score_i = model.gamma.item() - score_i
                            score_i += filter_bias
                            argsort_i = torch.argsort(score_i, dim = 1, descending=True)
                            argsort_list.append(argsort_i) 

                        score2 = model.gamma.item() - score2
                        score2 += filter_bias
                        argsort2 = torch.argsort(score2, dim = 1, descending=True)
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            ranking_min = 999999
                            ranking_max = -1
                            for argsort_i in argsort_list:
                                ranking_i = (argsort_i[i, :] == positive_arg[i]).nonzero()
                                if ranking_i < ranking_min:
                                    ranking_min = ranking_i
                                if ranking_i > ranking_max:
                                    ranking_max = ranking_i

                            assert ranking2.size(0) == 1
                            ranking2 = 1 + ranking2.item()
                            ranking_min = 1 + ranking_min.item()
                            ranking_max = 1 + ranking_max.item()

                            logs.append({
                                'MRR': 1.0/ranking_max,
                                'MR': float(ranking_max),
                                'HITS@1': 1.0 if ranking_max <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking_max <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking_max <= 10 else 0.0,
                            })
                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                        step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics


if __name__ == "__main__":
    pass
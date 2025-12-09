# ####################### fb15k-237 ######################################################
 CUDA_VISIBLE_DEVICES=0 python train_kgdm.py --cuda --dataset FB15K-237 --do_train --do_valid --do_test \
  --data_path ./data/FB15K-237  -b 512 -d 400 -g 18.0 \
  -a 0.5 -adv --modelconfig "./model_configs/Denoiser_Net.yaml"\
  -lr 0.00008 --max_steps 250000 --dataset_neg -n 256\
  -save models/FB15K-237-Conv --test_batch_size 8 \
#   --dataset_onehot


# ####################### wn18rr ######################################################
 CUDA_VISIBLE_DEVICES=0 python train_kgdm.py --cuda --dataset wn18rr --do_train --do_valid --do_test \
  --data_path ../data/wn18rr  -b 256 -d 1000 -g 10.0 \
  -a 0.5 -adv --modelconfig "../model_configs/Denoiser_Net.yaml" \
  -lr 0.00008 --max_steps 250000 --dataset_neg -n 128\
  -save '../models/wn18rr-Tnet' --test_batch_size 2 


# ####################### umls ######################################################
 CUDA_VISIBLE_DEVICES=0 python train_kgdm.py --cuda --dataset umls --do_train --do_valid --do_test  \
  --data_path ../data/umls  -b 512 -d 4000 -g 14.0 \
  -a 0.5 -adv --modelconfig "../model_configs/Denoiser_Net.yaml" \
  -lr 0.00005 --max_steps 15000 --dataset_neg -n 256\
  -save '../models/umls-Tnet' --test_batch_size 8 


# # ####################### kinship ######################################################
 CUDA_VISIBLE_DEVICES=0 python train_kgdm.py --cuda --dataset kinship --do_train --do_valid --do_test \
  --data_path ../data/kinship  -b 256 -d 4000 -g 10.0 \
  -a 0.5 -adv --modelconfig "../model_configs/Denoiser_Net.yaml" \
  -lr 0.00008 --max_steps 10000 --dataset_neg -n 128\
  -save '../models/kinship-Tnet' --test_batch_size 5 
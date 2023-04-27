CUDA_VISIBLE_DEVICES=3 python src/train.py \
  --dataset-name UHD \
  --train-dir ./data/UHD-LL/training_set/\
  --valid-dir ./data/UHD-LL/testing_set/ \
  --ckpt-save-path ./ckpts_training/ \
  --nb-epochs 1000 \
  --batch-size 2\
  --train-size 512 512 \
  --plot-stats \
  --cuda    

# CUDA_VISIBLE_DEVICES=3 python src/train.py \
#   --dataset-name LOLv1 \
#   --train-dir ./data/LOL-v1/our485/\
#   --valid-dir ./data/LOL-v1/eval15/ \
#   --ckpt-save-path ./ckpts_training/ \
#   --nb-epochs 1000 \
#   --batch-size 2\
#   --train-size 400 600 \
#   --plot-stats \
#   --cuda    

# CUDA_VISIBLE_DEVICES=3 python src/train.py \
#   --dataset-name LOLv2 \
#   --train-dir ./data/LOL-v2/train/\
#   --valid-dir ./data/LOL-v2/test/ \
#   --ckpt-save-path ./ckpts_training/ \
#   --nb-epochs 1000 \
#   --batch-size 2\
#   --train-size 400 600 \
#   --plot-stats \
#   --cuda    

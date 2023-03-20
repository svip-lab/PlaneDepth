CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train.py \
--png \
--model_name exp1 \
--use_denseaspp \
--use_mixture_loss \
--plane_residual \
--flip_right
CUDA_VISIBLE_DEVICES=0 python evaluate_depth_HR.py \
--eval_stereo \
--load_weights_folder ./log/ResNet/exp1_sd/best_models \
--models_to_load encoder depth \
--use_denseaspp \
--plane_residual \
--use_mixture_loss \
--batch_size 1 \
--width 1280 \
--height 384

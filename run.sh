python train.py --pretreained_model unilm --pretrained_model_path ./speedymind_ckpts --root_data_dir ./data/speedy_data/ \
                --news_attributes title --num_hidden_layers 8 --world_size 4 --lr 1e-4 --pretrain_lr 8e-6 --warmup True \
                --schedule_step 240000 --warmup_step 1000 --batch_size 42 --npratio 4 --beta_for_cache 0.002 --max_step_in_cache 2 \
                --savename ffdensmoe_glove_noLN_relu_22200 --news_dim 400 --model_dir /workspaceblobstore/v-wenjunpeng/mind/saved_models
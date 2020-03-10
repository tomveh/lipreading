python src/train.py\
       --batch_size 64\
       --data_root "/work/t405/T40511/work/vehvilt2/"\
       --easy 0\
       --learning_rate 1e-4\
       --weight_decay 1e-3\
       --d_model 512\
       --description "try larger lr but with fixed seq len"\
       --frontend_weights '/u/46/vehvilt2/unix/thesis/lipreading/lightning_logs/pretrain/version_3/weights/frontend_weights.pt'\
       --weight_hist 0\
       --track_grad_norm 0\
       --accumulate_grad_batches 1\
       --workers 12\
       --seq_inc_interval 0

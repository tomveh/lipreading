python src/train.py\
       --batch_size 100\
       --accumulate_grad_batches 1\
       --data_root "/work/t405/T40511/work/vehvilt2/"\
       --easy 0\
       --learning_rate 1e-4\
       --weight_decay 1e-3\
       --d_model 512\
       --description "test lrs2 on frontend that has been trained on lrw w/ bs 64 but now with also greedy decoding and some minor changes made earlier + no sched because it looks at CER -- run seq len 2 forever"\
       --weight_hist 0\
       --track_grad_norm 0\
       --workers 12\
       --seq_inc_interval 0\
       --min_epochs 1000\
       --max_epochs 1000

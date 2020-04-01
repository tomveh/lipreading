python src/pretrain.py\
       --resnet resnet18\
       --dataset lrw1\
       --learning_rate 1e-3\
       --weight_decay 1e-3\
       --batch_size 32\
       --data_root "/work/t405/T40511/work/vehvilt2/lrw1"\
       --workers 12\
       --weight_hist 0\
       --epochs 10\
       --description "Run pretrain with OneCycle"\
       --fast_dev_run 0

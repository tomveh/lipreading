python src/pretrain.py\
       --resnet resnet18\
       --dataset pretrain\
       --learning_rate 1e-4\
       --weight_decay 1e-3\
       --batch_size 32\
       --data_root "/work/t405/T40511/work/vehvilt2/"\
       --workers 16\
       --weight_hist 0\
       --min_epochs 10\
       --max_epochs 10\
       --description ""\
       --fast_dev_run 0

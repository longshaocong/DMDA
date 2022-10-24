python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/05/art_painting \
                --test_envs 0 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.5 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ./train_output/05/cartoon \
                --test_envs 1 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.5 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/05/photo \
                --test_envs 2 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.5 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/05/sketch \
                --test_envs 3 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.5 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/06/art_painting \
                --test_envs 0 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.6 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/06/cartoon \
                --test_envs 1 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.6 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/06/photo \
                --test_envs 2 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.6 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/06/sketch \
                --test_envs 3 \
                --dataset PACS \
                --algorithm JDM_con \
                --alpha 0.6 \
                --temperature 1 \
                --CON_lambda 0.01 \
                --batch_size 128 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 
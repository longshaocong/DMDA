# python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
#                 --max_epoch 100 \
#                 --seed 0 \
#                 --weight_decay 0.0001 \
#                 --net resnet18 \
#                 --output ./train_output/03/art_painting \
#                 --test_envs 0 \
#                 --dataset PACS \
#                 --algorithm CONTRA \
#                 --CON_lambda 0.3 \
#                 --batch_size 128 \
#                 --lr 0.025 \
#                 --contrast \
#                 --schuse \
#                 --schusech cos 

# python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
#                 --max_epoch 100 \
#                 --seed 0 \
#                 --weight_decay 0.0001 \
#                 --net resnet18 \
#                 --output ./train_output/03/cartoon \
#                 --test_envs 1 \
#                 --dataset PACS \
#                 --algorithm CONTRA \
#                 --CON_lambda 0.3 \
#                 --batch_size 128 \
#                 --lr 0.025 \
#                 --contrast \
#                 --schuse \
#                 --schusech cos 

# python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
#                 --max_epoch 100 \
#                 --seed 0 \
#                 --weight_decay 0.0001 \
#                 --net resnet18 \
#                 --output ../train_output/03/photo \
#                 --test_envs 2 \
#                 --dataset PACS \
#                 --algorithm CONTRA \
#                 --CON_lambda 0.3 \
#                 --batch_size 128 \
#                 --N_WORKERS 2 \
#                 --lr 0.025 \
#                 --contrast \
#                 --schuse \
#                 --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/03/sketch \
                --test_envs 3 \
                --dataset PACS \
                --algorithm CONTRA \
                --CON_lambda 0.3 \
                --batch_size 128 \
                --N_WORKERS 2 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/003/art_painting \
                --test_envs 0 \
                --dataset PACS \
                --algorithm CONTRA \
                --CON_lambda 0.03 \
                --batch_size 128 \
                --N_WORKERS 2 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/003/cartoon \
                --test_envs 1 \
                --dataset PACS \
                --algorithm CONTRA \
                --CON_lambda 0.03 \
                --batch_size 128 \
                --N_WORKERS 2 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/003/photo \
                --test_envs 2 \
                --dataset PACS \
                --algorithm CONTRA \
                --CON_lambda 0.03 \
                --batch_size 128 \
                --N_WORKERS 2 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --weight_decay 0.0001 \
                --net resnet18 \
                --output ../train_output/003/sketch \
                --test_envs 3 \
                --dataset PACS \
                --algorithm CONTRA \
                --CON_lambda 0.03 \
                --batch_size 128 \
                --N_WORKERS 2 \
                --lr 0.025 \
                --contrast \
                --schuse \
                --schusech cos 
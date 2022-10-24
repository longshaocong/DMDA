python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/07/art_painting \
                --test_envs 0 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.7 \
                --batch_size 32 \
                --lr 1e-3 \

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/07/cartoon \
                --test_envs 1 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.7 \
                --batch_size 32 \
                --lr 1e-3 \

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/07/photo \
                --test_envs 2 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.7 \
                --batch_size 32 \
                --lr 1e-3 \


python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/07/sketch \
                --test_envs 3 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.7 \
                --batch_size 32 \
                --lr 1e-3 \

python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/08/art_painting \
                --test_envs 0 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.8 \
                --batch_size 32 \
                --lr 1e-3 \


python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/08/cartoon \
                --test_envs 1 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.8 \
                --batch_size 32 \
                --lr 1e-3 \


python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/08/photo \
                --test_envs 2 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.8 \
                --batch_size 32 \
                --lr 1e-3 \


python train.py --data_dir /openbayes/home/DomainBed/domainbed/data/PACS/ \
                --max_epoch 100 \
                --seed 0 \
                --temperature 4 \
                --net resnet50 \
                --output ../train_output/Res50/4/08/sketch \
                --test_envs 3 \
                --dataset PACS \
                --algorithm JDM \
                --vary_T \
                --alpha 0.8 \
                --batch_size 32 \
                --lr 1e-3 \
  

python train.py --data_dir ../DG_via_Joint/transferlearning/data/PACS/ \
                --max_epoch 100 \
                --net resnet18 \
                --output ./train_output/0505/art_painting \
                --test_envs 0 \
                --dataset PACS \
                --algorithm JDM \
                --alpha 0.5 \
                --batch_size 32 \
                --lr 1e-3
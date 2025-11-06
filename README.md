# Rethinking Domain Generalization: Discriminability and Generalizability

Welcome to the repository for our paper: "Rethinking Domain Generalization: Discriminability and Generalizability."

## Installation
### Environments

Environment details used for the main experiments. 
```
Environment:
	Python: 3.7.13
	PyTorch: 1.12.1
	Torchvision: 0.13.1
	CUDA: 10.2
	CUDNN: 7605
	NumPy: 1.21.5
	PIL: 9.5.0
```

### Dependencies

```sh
pip install -r requirements.txt
```


## Usage

### Training

#### Training on single node

You can use the following training command to train DMDA.
We provide the sample on PACS with 'Art painting' as the target domain.

```bash
python train.py --data_dir your_data_path 
                --max_epoch 50 
                --temperature 4 
                --net resnet50 
                --output ./train_output 
                --test_envs 0 
                --dataset PACS 
                --algorithm DMDA 
                --batch_size 32 
                --lr learning_rate 
                --weight_decay 5e-4 
                --alpha alpha
                --ratio ratio
                --steps_per_epoch 100 
```


## Citation

If you find DMDA useful in your research, please consider citing:
```bibtex
@article{long2024rethinking,
  title={Rethinking domain generalization: Discriminability and generalizability},
  author={Long, Shaocong and Zhou, Qianyu and Ying, Chenhao and Ma, Lizhuang and Luo, Yuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={34},
  number={11},
  pages={11783--11797},
  year={2024}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.
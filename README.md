## Convolutional Spectral Kernel Learning with Generalization Guarantees

## Intro
This repository provides the code used to run the experiments of the paper "Convolutional Spectral Kernel Learning with Generalization Guarantees".
## Environments
- Python 3.7.4
- Pytorch 1.10.0
- NNI 2.5
- CUDA 10.1.168
- cuDnn 7.6.0
- Gpytorch 1.6.0
- GPU: Nvidia RTX 2080Ti 11G
## Core functions
- `models.py` implements random features based neural networks, including RFFNet and DSKN, their convoluational versions CRFFNet and CSKN, and their vgg-type versions CRFFNet8 and CSKN8. Meanwhile, models.py implements the `train()` and `test()` methods for these models.
- `utils.py` implements useful tools for loading primal and resized datasets to feed neural networks. `densenet.py` is used to implement deep kernel learning (DKL) from [Gpytorch](https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html).
- `exp_ablation0_*.py`, `exp_ablation1_layers.py`, `exp_ablation2_regularizers.py`, and `exp_ablation3_convolutional.py` are used to explore the effects of non-stationary spectral kernels, the depth of network, additional regularizers, and convolutional filters on the MNIST dataset, respectively.
- `exp_comparison_CSKN.py` trains `CSKN8` and `CRFFNet8` on large image datasets, while `exp_comparison_DKLModel.py` and `exp_comparison_PretrainedModels.py` implement and train DKL and popular CNN networks, respectively. 
- `ipy_exp_ablation.ipynb` and `ipy_exp_comparison.ipynb` load and illustrate ablation and comparison experimental results, respectively.
- `tools` folder is used to tune hyperparameters vi [NNI](https://github.com/microsoft/nni) and estimating the spectral density.
`data` folder stores all datasets, including `MNIST, FashionMNIST, CIFAR10, CIFAR100`, and `TinyImagenet`.
`results` folder records experimental results and `figures` folder stores pdf files for illustrating experimental results.

## Experiments
1. Run the following scripts for ablation expriments
```
python3 exp_ablation0_RFFNet.py
python3 exp_ablation0_DSKN.py
python3 exp_ablation1_layers.py --num_layers 1
python3 exp_ablation1_layers.py --num_layers 2
python3 exp_ablation1_layers.py --num_layers 3
python3 exp_ablation1_layers.py --num_layers 4
python3 exp_ablation1_layers.py --num_layers 5
python3 exp_ablation2_regularizers.py
python3 exp_ablation3_convolutional.py --kernel_size 3
python3 exp_ablation3_convolutional.py --kernel_size 5
python3 exp_ablation3_convolutional.py --kernel_size 7
```
and then run codes in `ipy_exp_ablation.ipynb` to illstrate experimental results.

2. Download [`TinyImagenet` dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip ) into `data` folder and process it by `tinyimagenet_preprocess.py`. 
The other datasets can be downloaded automatically when they are used for the first time.

3. Run comparison expriments on datatsets `MNIST, FashionMNIST, CIFAR10, CIFAR100, TinyImagenet` for `CSKN8` and `CRFFNet8`
```
python3 exp_comparison_CSKN.py --model CSKN8 --dataset $dataset_name --epochs 300 --repeates 1
python3 exp_comparison_CSKN.py --model CRFFNet8 --dataset $dataset_name --epochs 300 --repeates 1
```
for `DKL` models
```
python3 exp_comparison_DKLModel.py --dataset $dataset_name --epochs 300 --repeates 1
```
and for popular CNN models `resnet, vgg, densenet, shufflenet`
```
python3 exp_comparison_PretrainedModels.py --model resnet --dataset $dataset_name --epochs 300 --repeates 1
python3 exp_comparison_PretrainedModels.py --model vgg --dataset $dataset_name --epochs 300 --repeates 1
python3 exp_comparison_PretrainedModels.py --model densenet --dataset $dataset_name --epochs 300 --repeates 1
python3 exp_comparison_PretrainedModels.py --model shufflenet --dataset $dataset_name --epochs 300 --repeates 1
```
Run ipy_exp_comparison.ipynb to draw figures

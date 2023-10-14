# Learning to (Learn at Test Time)

## GPU Setup

We first discuss how to set up and run our code on a (local) GPU machine. 
We highly recommend using a virtual environment when installing python dependencies.

### Install CUDA environment

Let's first download [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn), 
which are needed by JAX (gpu version). We would have the following file structure:
```
/YOUR/CUDA/PATH
├── bin
├── include
├── lib64
    ...
```
```
/YOUR/cuDNN/PATH
├── LICENSE
├── include
└── lib
```
and we can copy the files in ```/YOUR/cuDNN/PATH/include``` and ```/YOUR/cuDNN/PATH/lib``` to 
```/YOUR/CUDA/PATH/include``` and ```/YOUR/CUDA/PATH/lib64``` respectively.

Then we export the following environment variables:
```
export CUDA_HOME=/YOUR/CUDA/PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
```

### Install Python packages

```
git clone https://github.com/LeoXinhaoLee/MTTT.git
cd MTTT
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Then install the latest version of JAX library as
```
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
You may need a different JAX package, depending on CUDA and cuDNN libraries installed on your machine. 
Please consult [official JAX documentation](https://github.com/google/jax#pip-installation-gpu-cuda) for more information.

### Prepare TFDS data

1. Manually create a directory with the following structure (or use $TFDS_DATA_DIR if exists, which is default to ~/tensorflow_datasets/):
```
/YOUR/TFDS/PATH
├── downloads
   └── manual
```
2. Manually [download imagenet2012](https://www.image-net.org/challenges/LSVRC/2012/).
3. Place the downloaded ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar files under ```YOUR/TFDS/PATH/downloads/manual```. 
4. Run the following commands (which may take ~1 hour):
```
cd MTTT
export TFDS_DATA_DIR=/YOUR/TFDS/PATH
python3 ./tools/download_tfds_datasets.py imagenet2012
``` 

After the above procedure, the dataset should have the following structure:
```
/YOUR/TFDS/PATH
├── downloads
│   ├── extracted
│       ├── train
│       └── val
│   └── manual
│       ├── ILSVRC2012_img_train.tar
│       └── ILSVRC2012_img_val.tar
└── imagenet2012
│   └── 5.*.*
│       ├── dataset_info.json
│       ├── features.json
│       ├── imagenet2012-train.tfrecord-00000-of-01024
│       ├── imagenet2012-validation.tfrecord-00000-of-00064
              ...
```

## Cloud TPU VM Setup

### Create TPU VMs, prepare TFDS data
Please refer to [this link](https://github.com/google-research/big_vision/tree/main#preparing-tfds-data) for guidance on creating 
cloud TPU VMs. 
The TFDS data preparation is similar to that for GPU.

## Quick Start

Given a local machine with GPUs, the following command reproduces our MTTT MLP result (Acc=74.6%) on Patch ImageNet:
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_itr=1 \
                --config.inner.TTT.inner_lr='(1.,)' \
                --config.input.accum_time=1 \
                --workdir ./exp/patch_MTTT_MLP
```

To specify a custom path to your dataset, you could either code it in 
```config_patch.py``` and ```config_pixel.py```, or specify ```--config.tfds_path=/YOUR/TFDS/PATH``` 
when launching a job.

Please note that you may need to increase ```--config.input.accum_time```  to accommodate the memory constraint of your GPU devices.

Experiment statistics (train/val top-1 accuracy, loss, inner loss) will be saved in ```./exp/patch_MTTT_MLP/all_stat_dict.pth```, 
and model and optimizer state checkpoint will be automatically saved in ```./ckpt/patch_MTTT_MLP/checkpoint.npz```.

## Plot Learning Curves

We provide two simple files for plotting the curve of validation error and MTTT inner loss throughout training.

### Validation error curve
Fill in the ```folder_names``` list in ```plot_multi.py``` with the names of experiment folders you want to plot, 
and then run 
```
python plot_multi.py
```

Our tool supports the comparison of validation error curves of multiple experiments at once.

### MTTT inner loss curve
Set the ```folder_name``` variable in ```plot_inner.py``` to the name of the experiemnt you want to plot, and 
then run 
```
python plot_inner.py
```

Our tool currently only supports plotting one experiment at a time. Besides, the concept of MTTT inner loss only applies to MTTT, 
thus we manually set inner loss of self-attention and linear attention to ```inf``` to ensure it's not meaningful.

## Commands for All Experiments

### Patch ImageNet

For Patch ImageNet, the following commands use ViT-Small by default. You may add ```--config.model=tiny``` to use ViT-Tiny, 
or customize our code for other models.

MTTT MLP (itr=1): 
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_itr=1 \
                --config.inner.TTT.inner_lr='(1.,)' \
                --workdir ./exp/patch_MTTT_MLP
```

MTTT MLP (itr=4): 
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_itr=4 \
                --config.inner.TTT.inner_lr='(1.,1.,1.,1.)' \
                --workdir ./exp/patch_MTTT_MLP_itr=4
```

MTTT Linear (equivalent to Linear Attention):
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_encoder=mlp_1 \
                --config.inner.TTT.inner_encoder_init=zero \
                --config.inner.TTT.inner_encoder_bias=False \
                --config.inner.TTT.decoder_LN=False \
                --config.inner.TTT.train_init=False \
                --workdir ./exp/patch_MTTT_linear
```

Linear Attention (equivalent to MTTT Linear):
```
python train.py --config config_patch.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=False \
                --config.inner.linear_attention.normalizer=constant \
                --workdir ./exp/patch_linear_attention
```

Linear Attention (Katharopoulos et al.):
```
python train.py --config config_patch.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=True \
                --config.inner.linear_attention.normalizer=adaptive \
                --workdir ./exp/patch_linear_attention_Katharopoulos
```

Self-Attention (Softmax-Attention):
```
python train.py --config config_patch.py \
                --config.inner.layer_type=self_attention \
                --workdir ./exp/patch_self_attention
```

### Pixel ImageNet

For Pixel ImageNet, the following commands use ViT-Tiny by default. You may add ```--config.model=small``` to use ViT-Small, 
or customize our code for other models.

MTTT MLP (itr=1): 
```
python train.py --config config_pixel.py \
                --config.inner.TTT.inner_itr=1 \
                --config.inner.TTT.inner_lr='(1.,)' \
                --config.inner.TTT.SGD=False \
                --workdir ./exp/pixel_MTTT_MLP
```

MTTT MLP (itr=4, SGD): 
```
python train.py --config config_pixel.py \
                --config.inner.TTT.inner_itr=4 \
                --config.inner.TTT.inner_lr='(1.,1.,1.,1.)' \
                --config.inner.TTT.SGD=True \
                --workdir ./exp/pixel_MTTT_MLP_itr=4_SGD
```

MTTT Linear (equivalent to Linear Attention):
```
python train.py --config config_pixel.py \
                --config.inner.TTT.inner_encoder=mlp_1 \
                --config.inner.TTT.inner_encoder_init=zero \
                --config.inner.TTT.inner_encoder_bias=False \
                --config.inner.TTT.decoder_LN=False \
                --config.inner.TTT.train_init=False \
                --workdir ./exp/pixel_MTTT_linear
```

Linear Attention (equivalent to MTTT Linear):
```
python train.py --config config_pixel.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=False \
                --config.inner.linear_attention.normalizer=constant \
                --workdir ./exp/pixel_linear_attention
```

Linear Attention (Katharopoulos et al.):
```
python train.py --config config_pixel.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=True \
                --config.inner.linear_attention.normalizer=adaptive \
                --workdir ./exp/pixel_linear_attention_Katharopoulos
```

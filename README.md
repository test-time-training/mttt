# Learning to (Learn at Test Time)

[**GPU Setup**](#gpu-setup)
| [**Cloud TPU VM Setup**](#cloud-tpu-vm-setup)
| [**Quick Start**](#quick-start)
| [**Plot Statistics**](#plot-statistics)
| [**Commands for All Experiments**](#commands-for-all-experiments)

## GPU Setup

To setup and run our code on a (local) GPU machine, we highly recommend using a virtual environment when installing python dependencies.

### Install CUDA environment

First download [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn), 
required by JAX (the gpu version). 
Please use the following file structure:
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
and copy the files in ```/YOUR/cuDNN/PATH/include``` and ```/YOUR/cuDNN/PATH/lib``` to 
```/YOUR/CUDA/PATH/include``` and ```/YOUR/CUDA/PATH/lib64``` respectively.

Next, export the following environment variables:
```
export CUDA_HOME=/YOUR/CUDA/PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
```

### Install Python packages
First run the following commands

```
git clone https://github.com/LeoXinhaoLee/MTTT.git
cd MTTT
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Then install the latest JAX library
```
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
You may need a different JAX package, depending on the version of CUDA and cuDNN libraries installed on your machine. 
Please consult the [official JAX documentation](https://github.com/google/jax#pip-installation-gpu-cuda) for more information.

### Prepare TFDS data

1. Manually create a directory with the following structure:
```
/YOUR/TFDS/PATH
├── downloads
   └── manual
```
Or you can use ```$TFDS_DATA_DIR``` if it exists. This should be ```~/tensorflow_datasets/``` by default.</br>
2. Download [imagenet2012](https://www.image-net.org/challenges/LSVRC/2012/).</br>
3. Place the downloaded files ```ILSVRC2012_img_train.tar``` and ```ILSVRC2012_img_val.tar``` under ```YOUR/TFDS/PATH/downloads/manual```. </br>
4. Run the following commands (which may take ~1 hour):
```
cd MTTT
export TFDS_DATA_DIR=/YOUR/TFDS/PATH
python3 ./tools/download_tfds_datasets.py imagenet2012
``` 

After the above procedure, you should have the following file structure:
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
Please refer to [this link](https://github.com/google-research/big_vision/tree/main#preparing-tfds-data) for guidance on creating cloud TPU VMs. 
TFDS data preparation is similar to that for GPU.

## Quick Start

Given a local machine with GPUs, the following command reproduces our result for MTTT-MLP (accurary 74.6%) on ImageNet (from patches):
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_itr=1 \
                --config.inner.TTT.inner_lr='(1.,)' \
                --config.input.accum_time=1 \
                --workdir ./exp/patch_MTTT_MLP
```

To specify a custom path to your dataset, you could either modify its assignment in
```config_patch.py``` and ```config_pixel.py```, or specify ```--config.tfds_path=/YOUR/TFDS/PATH``` 
when launching a job.

To accommodate the memory constraint of your devices, you may need to tune the number of gradient accumulation steps by specifying ```--config.input.accum_time```.

Statistics for the experiment will be saved in ```./exp/patch_MTTT_MLP/all_stat_dict.pth```. These include traing and validation accuracy and loss, as well as inner-loop reconstruction loss.
The most recent checkpoint for model and optimizer state will be saved in ```./ckpt/patch_MTTT_MLP/checkpoint.npz```.

## Plot Statistics

We provide two simple files to plot the saved statistics during training.

### Validation error
Fill the ```folder_names``` list in ```plot_multi.py``` with the folder names of the experiments you want to plot together, then run 
```
python plot_multi.py
```

It can compare the learning curves of multiple experiments at once.

### Inner-loop reconstruction loss
Set the ```folder_name``` variable in ```plot_inner.py``` to the name of the experiment you want to plot, 
then run 
```
python plot_inner.py
```

It plots one experiment at a time, for all TTT layers together (12 by default).
Since the concept of inner-loop reconstruction loss only applies to MTTT, 
we set this loss for self-attention and linear attention to ```inf``` to signal that it is not meaningful.

## Commands for All Experiments

### ImageNet (from patches)

For ImageNet (from patches), all commands use ViT-Small by default.

As noted above, you may need to tune the number of gradient accumulation steps by specifying ```--config.input.accum_time``` to accommodate the memory constraint of your devices.

MTTT-MLP (SGD T=1, i.e. no SGD): 
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_itr=1 \
                --config.inner.TTT.inner_lr='(1.,)' \
                --workdir ./exp/patch_MTTT_MLP
```

MTTT-MLP SGD T=4: 
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_itr=4 \
                --config.inner.TTT.inner_lr='(1.,1.,1.,1.)' \
                --workdir ./exp/patch_MTTT_MLP_itr=4
```

MTTT-Linear:
```
python train.py --config config_patch.py \
                --config.inner.TTT.inner_encoder=mlp_1 \
                --config.inner.TTT.inner_encoder_init=zero \
                --config.inner.TTT.inner_encoder_bias=False \
                --config.inner.TTT.decoder_LN=False \
                --config.inner.TTT.train_init=False \
                --workdir ./exp/patch_MTTT_linear
```

Linear attention:
```
python train.py --config config_patch.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=False \
                --config.inner.linear_attention.normalizer=constant \
                --workdir ./exp/patch_linear_attention
```

Linear attention (Katharopoulos et al.):
```
python train.py --config config_patch.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=True \
                --config.inner.linear_attention.normalizer=adaptive \
                --workdir ./exp/patch_linear_attention_Katharopoulos
```

Self-attention (with softmax):
```
python train.py --config config_patch.py \
                --config.inner.layer_type=self_attention \
                --workdir ./exp/patch_self_attention
```

### ImageNet from pixels

For ImageNet from pixels, the following commands use ViT-Tiny by default. You may add ```--config.model=small``` to use ViT-Small. 

Each run can take at least a few days on most machines.

As noted above, you may need to tune the number of gradient accumulation steps by specifying ```--config.input.accum_time``` to accommodate the memory constraint of your devices.


MTTT-MLP (SGD T=1, i.e. no SGD): 
```
python train.py --config config_pixel.py \
                --config.inner.TTT.inner_itr=1 \
                --config.inner.TTT.inner_lr='(1.,)' \
                --config.inner.TTT.SGD=False \
                --workdir ./exp/pixel_MTTT_MLP
```

MTTT-MLP SGD T=4: 
```
python train.py --config config_pixel.py \
                --config.inner.TTT.inner_itr=4 \
                --config.inner.TTT.inner_lr='(1.,1.,1.,1.)' \
                --config.inner.TTT.SGD=True \
                --workdir ./exp/pixel_MTTT_MLP_itr=4_SGD
```

MTTT-Linear:
```
python train.py --config config_pixel.py \
                --config.inner.TTT.inner_encoder=mlp_1 \
                --config.inner.TTT.inner_encoder_init=zero \
                --config.inner.TTT.inner_encoder_bias=False \
                --config.inner.TTT.decoder_LN=False \
                --config.inner.TTT.train_init=False \
                --workdir ./exp/pixel_MTTT_linear
```

Linear attention:
```
python train.py --config config_pixel.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=False \
                --config.inner.linear_attention.normalizer=constant \
                --workdir ./exp/pixel_linear_attention
```

Linear attention (Katharopoulos et al.):
```
python train.py --config config_pixel.py \
                --config.inner.layer_type=linear_attention \
                --config.inner.linear_attention.elu=True \
                --config.inner.linear_attention.normalizer=adaptive \
                --workdir ./exp/pixel_linear_attention_Katharopoulos
```

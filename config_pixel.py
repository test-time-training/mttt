import ml_collections as mlc
import os


def get_config(arg=None):
  config = mlc.ConfigDict()
  
  config.benchmark = "pixel"

  ###
  # Inner Loop
  ###
  config.inner = dict()
  config.inner.layer_type = 'TTT'  # 'TTT' | 'self_attention' | 'linear_attention'

  # Linear Attention
  config.inner.linear_attention = dict()
  config.inner.linear_attention.elu = True
  config.inner.linear_attention.normalizer = 'adaptive'  # 'adaptive' | 'constant'

  # TTT Layer
  config.inner.TTT = dict()
  config.inner.TTT.inner_encoder = "mlp_2"
  config.inner.TTT.inner_itr = 1
  config.inner.TTT.inner_lr = (1.,)
  config.inner.TTT.train_init = True
  config.inner.TTT.inner_encoder_init = "xavier_uniform"
  config.inner.TTT.inner_encoder_bias = True
  config.inner.TTT.decoder_LN = True
  config.inner.TTT.SGD = True  # For pixel benchmark, use SGD for inner optimization
  ###

  ###
  # Routine & Logging
  ###
  config.total_epochs = 90
  config.resume = False
  ###

  ###
  # Optimizer & Scheduler
  ###
  config.grad_clip_norm = 1.0
  config.optax_name = "scale_by_adam"
  config.optax = dict(mu_dtype="bfloat16")

  config.lr = 0.001
  config.wd = 0.0001
  config.wd_mults = [(".*/kernel$", 1.0)]
  ###

  ###
  # Common
  ###
  config.seed = 0
  config.tf_seed = 0
  config.num_classes = 1000
  config.loss = "softmax_xent"

  config.model = "tiny"  # "tiny" | "small"

  config.tfds_path = ""  # TODO: Change to your custom data path

  config.input = {}
  config.input.data = dict(
    name="imagenet2012",
    split="train",
    data_dir=config.tfds_path,
  )
  config.input.batch_size = 1024
  config.input.accum_time = 2  # TODO: Pixel benchmark needs more grad accum times (e.g. 2 on 512-pod)
  config.input.cache_raw = True  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000
  config.mixup = dict(p=0, fold_in=None)

  config.pp_common = (
    "|value_range(-1, 1)"
    "|onehot(1000, key='label', key_result='labels')"
    "|keep('image', 'labels')"
  )

  config.input.pp = (
                     "decode_jpeg_and_inception_crop(224)"
                     ) + config.pp_common
  pp_eval = (
             "decode"
             "|resize_small(256)"
             "|central_crop(224)"
             ) + config.pp_common

  config.evals = {
    "type": "classification",
    "data": dict(name="imagenet2012",
                 split="validation",
                 data_dir=config.tfds_path,
                 ),
    "pp_fn": pp_eval,
    "loss_name": config.loss,
  }
  ###

  return config
    
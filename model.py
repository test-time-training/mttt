"""
This code is adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
"""

from typing import Optional, Any, Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from inner_loop import *


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  y, x = jnp.mgrid[:h, :w]
  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError("Unknown posemb type: %s" % typ)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim

  @nn.compact
  def __call__(self, x):
    """Apply Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )
    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (TTT Layer + MLP)."""
  width: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 6
  config: Any = None

  @nn.compact
  def __call__(self, x):
    B, N, d = x.shape  # pylint: disable=unused-variable

    y = nn.LayerNorm()(x)
    if self.config.layer_type == "self_attention":
      y = SelfAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=True,
      )(y, y)
      inner_loss_tuple = (jnp.inf, jnp.inf)  # inner loss only applies to TTT Layers
    elif self.config.layer_type == "linear_attention":
      y = LinearAttention(
        width=self.width,
        num_heads=self.num_heads,
        config=self.config.linear_attention,
      )(y)
      inner_loss_tuple = (jnp.inf, jnp.inf)  # inner loss only applies to TTT Layers
    elif self.config.layer_type == "TTT":
      y, inner_loss_tuple = TTTLayer(width=self.width,
                                     num_heads=self.num_heads,
                                     config=self.config.TTT)(y)
    else:
      raise NotImplementedError("Layer Type %s Not Implemented." % (self.config.layer_type))
    x = x + y

    y = nn.LayerNorm()(x)
    y = MlpBlock(mlp_dim=self.mlp_dim)(y)
    x = x + y

    return x, inner_loss_tuple


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  width: int
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  config: Any = None

  @nn.compact
  def __call__(self, x):
    inner_loss_tuple_layers = ()
    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          width=self.width, mlp_dim=self.mlp_dim, num_heads=self.num_heads,
          config=self.config)
      x, inner_loss_tuple = block(x)
      inner_loss_tuple_layers += (inner_loss_tuple,)

    return nn.LayerNorm(name="encoder_norm")(x), inner_loss_tuple_layers


class Model(nn.Module):
  width: int
  depth: int
  mlp_dim: int
  num_heads: int
  num_classes: int = 1000
  patch_size: Sequence[int] = (16, 16)
  posemb: str = "sincos2d"
  head_zeroinit: bool = True
  config: Any = None

  def setup(self) -> None:
    self.word_embeddings = nn.Conv(
      features=self.width,
      kernel_size=self.patch_size, 
      strides=self.patch_size,
      padding="VALID",
      param_dtype=jnp.float32,
      name="embedding")

    self.pos_emb = get_posemb(
                   self, self.posemb, (224 // self.patch_size[0], 224 // self.patch_size[1]),
                   self.width, "pos_embedding", jnp.float32)

    self.encoder = Encoder(
      width=self.width,
      depth=self.depth,
      mlp_dim=self.mlp_dim,
      num_heads=self.num_heads,
      config=self.config,
      name="Transformer")

    self.pre_logit = nn.Dense(self.width, name="pre_logits")
    kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
    self.head = nn.Dense(self.num_classes, name="head", **kw)

  def __call__(self, image):
    B, H, W, C = image.shape

    tok_emb = self.word_embeddings(image)
    tok_emb = tok_emb.reshape(B, -1, self.width)

    x = tok_emb + self.pos_emb

    x, inner_loss_tuple_layers = self.encoder(x)

    x = jnp.mean(x, axis=1)
    x = nn.tanh(self.pre_logit(x))
    x = self.head(x)

    return x, inner_loss_tuple_layers

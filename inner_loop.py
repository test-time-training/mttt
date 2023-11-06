from typing import Any
from functools import partial

import jax
from jax import vmap
import flax.traverse_util as traverse_util
import flax.linen as nn
import jax.numpy as jnp


###
# Baseline Layers
###
SelfAttention = nn.MultiHeadDotProductAttention


class LinearAttention(nn.Module):
  """Linear Attention Layer (https://arxiv.org/abs/2006.16236)"""
  width: int
  num_heads: int
  config: Any = None

  def setup(self) -> None:
    initializer = jax.nn.initializers.xavier_uniform()
    self.value = nn.Dense(self.width, kernel_init=initializer)
    self.query = nn.Dense(self.width, kernel_init=initializer)
    self.key = nn.Dense(self.width, kernel_init=initializer)
    self.proj = nn.Dense(self.width, kernel_init=initializer)

  def split_head(self, hidden_states):
    hidden_states = hidden_states.reshape(*hidden_states.shape[:2],
                                          self.num_heads,
                                          hidden_states.shape[-1] // self.num_heads)
    hidden_states = jnp.einsum("...nhd->...hnd", hidden_states)
    return hidden_states

  def __call__(self, x):
    v = self.value(x)
    q = self.query(x)
    k = self.key(x)

    if self.config.elu:
      q = jax.nn.elu(q) + 1.
      k = jax.nn.elu(k) + 1.

    v, q, k = self.split_head(v), self.split_head(q), self.split_head(k)  # [B,H,N,d/H]

    k_v = jnp.einsum("...ki,...kj->...ij", k, v)
    numerator = jnp.einsum("...ik,...kj->...ij", q, k_v)

    if self.config.normalizer == 'adaptive':
      sum_k = k.sum(axis=-2, keepdims=True)
      denominator = jnp.einsum("...ik,...jk->...ij", q, sum_k)
    elif self.config.normalizer == 'constant':
      denominator = v.shape[-2] * v.shape[-1]  # normalizer = N * d / H
    else:
      raise NotImplementedError("Linear Attention Normalizer %s Not Implemented." % (self.config.normalizer))

    y = numerator / denominator
    y = jnp.einsum("...hnd->...nhd", y)
    y = y.reshape(*y.shape[:2], -1)
    y = self.proj(y)

    return y


###
# TTT Layer
###
class TTTEncoder(nn.Module):
  mlp_dim: int
  config: Any = None

  @nn.compact
  def __call__(self, x):
    if self.config.inner_encoder == "mlp_1":
      y = nn.Dense(self.mlp_dim, use_bias=self.config.inner_encoder_bias,
                   name="inner_Dense_0")(x)
    elif self.config.inner_encoder == "mlp_2":
      y = nn.Dense(int(self.mlp_dim * 4), use_bias=self.config.inner_encoder_bias,
                   name="inner_Dense_0")(x)
      y = nn.gelu(y)
      y = nn.Dense(self.mlp_dim, use_bias=self.config.inner_encoder_bias,
                   name="inner_Dense_1")(y)
    else:
      raise NotImplementedError("Inner Encoder %s Not Implemented." % (self.config.inner_encoder))

    return y


class DummyLinearLayer(nn.Module):
  width: int
  use_bias: bool
  name: str

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.width, use_bias=self.use_bias, name=self.name)(x)
    return x


class DummyLayerNorm(nn.Module):
  name: str

  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm(name=self.name)(x)
    return x


class DummyNoOp(nn.Module):
  @nn.compact
  def __call__(self, x):
    return x


class TTTLayer(nn.Module):
  width: int
  num_heads: int
  config: Any = None
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.psi = DummyLinearLayer(width=self.width // self.num_heads, use_bias=True, name="psi")
    psi_params = self.psi.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]

    self.phi = DummyLinearLayer(width=self.width // self.num_heads, use_bias=True, name="phi")
    phi_params = self.phi.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]

    self.g = DummyLinearLayer(width=self.width, use_bias=False, name="g")
    g_params = self.g.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]
    self.g_bias = self.param("g_bias", jax.nn.initializers.zeros, (1, self.width), self.dtype)

    self.h = DummyLinearLayer(width=self.width, use_bias=False, name="h")
    h_params = self.h.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]
    self.h_bias = self.param("h_bias", jax.nn.initializers.zeros, (1, self.width), self.dtype)

    self.encoder = TTTEncoder(mlp_dim=self.width // self.num_heads, config=self.config)
    encoder_params = self.encoder.init(jax.random.PRNGKey(0), jnp.ones([1, self.width // self.num_heads]))["params"]

    def get_multi_head_params(params, kernel_init="xavier"):
      flat_params = traverse_util.flatten_dict(params, sep="/")
      for k in flat_params.keys():
        if "kernel" in k:
          if kernel_init == "xavier_uniform":
            initializer = nn.initializers.xavier_uniform()
          elif kernel_init == "zero":
            initializer = nn.initializers.zeros
          elif kernel_init == "vs_fan_in":
            initializer = nn.initializers.variance_scaling(scale=1., mode="fan_in", distribution="uniform")
          elif kernel_init == "vs_fan_out":
            initializer = nn.initializers.variance_scaling(scale=1., mode="fan_out", distribution="uniform")
          else:
            raise NotImplementedError("Initializer %s Not Implemented." % (kernel_init))
          p = self.param(k, initializer, (self.num_heads, *flat_params[k].shape), self.dtype)
        elif 'scale' in k:
          # initialize scale to 1
          p = self.param(k, jax.nn.initializers.ones, (self.num_heads, *flat_params[k].shape), self.dtype)
        else:
          # initialize bias to 0
          p = self.param(k, jax.nn.initializers.zeros, (self.num_heads, *flat_params[k].shape), self.dtype)
        flat_params[k] = p
      params_init = traverse_util.unflatten_dict(flat_params, sep="/")
      return params_init

    self.encoder_params = get_multi_head_params(encoder_params, self.config.inner_encoder_init)
    self.psi_params = get_multi_head_params(psi_params, "vs_fan_in")
    self.phi_params = get_multi_head_params(phi_params, "vs_fan_in")
    self.g_params = get_multi_head_params(g_params, "vs_fan_out")
    self.h_params = get_multi_head_params(h_params, "vs_fan_out")

    if self.config.decoder_LN:
      self.decoder_LN = DummyLayerNorm()
      decoder_LN_params = self.decoder_LN.init(jax.random.PRNGKey(0), jnp.ones([1, self.width]))["params"]
    else:
      self.decoder_LN = DummyNoOp()
      decoder_LN_params = {}
    self.decoder_LN_params = get_multi_head_params(decoder_LN_params, "layer_norm")

  def __call__(self, batch):
    @partial(vmap)
    def update_embed(sequence):
      """
      vmap over B sequences
      sequence: [N,d]
      """
      def inner_loss_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head):
        inner_input = self.phi.apply({"params": phi_params}, sequence_head)
        inner_input_transformed = self.encoder.apply({"params": encoder_params}, inner_input)
        inner_output = self.g.apply({"params": g_params}, inner_input_transformed)
        inner_output = inner_output + self.g_bias
        inner_output = self.decoder_LN.apply({"params": decoder_LN_params}, inner_output)
        loss = 0.5 * ((inner_output - sequence_head) ** 2).mean() * self.num_heads  # normalizer = N * d / H
        return loss

      @partial(vmap, axis_name="head")
      def parallelize_over_heads(psi_params, phi_params, encoder_params, g_params, decoder_LN_params, h_params,
                                 sequence_head):
        """
        vmap over H heads
        """
        grad_fn = jax.value_and_grad(inner_loss_fn, argnums=1)

        ilr = jnp.asarray(self.config.inner_lr, dtype=jnp.float32)
        inner_loss_tuple = ()
        # TODO: To avoid OOM, manually copy inner iteration for up to 4 times
        if self.config.SGD:
          N = sequence_head.shape[0]
          shuffle_rng = self.make_rng("idx")
          shuffle_rng = jax.random.fold_in(shuffle_rng, jax.lax.axis_index("head"))
          noise = jax.random.uniform(shuffle_rng, (N,), jnp.float32)
          order = jnp.argsort(noise)
          batches = sequence_head[order].reshape(self.config.inner_itr, N // self.config.inner_itr, -1)

          if self.config.inner_itr >= 1:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[0])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[0] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 2:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[1])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[1] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 3:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[2])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[2] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 4:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, batches[3])
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[3] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

        else:
          if self.config.inner_itr >= 1:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[0] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 2:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[1] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 3:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[2] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

          if self.config.inner_itr >= 4:
            inner_loss, grad = grad_fn(phi_params, encoder_params, g_params, decoder_LN_params, sequence_head)
            encoder_params = jax.tree_util.tree_map(lambda p, g: p - ilr[3] * g, encoder_params, grad)
            inner_loss_tuple += (inner_loss,)

        encoder_params_new = encoder_params
        # TODO: For precise profiling, comment out the below 2 lines to avoid unnecessary compute
        inner_loss_final = inner_loss_fn(phi_params, encoder_params_new, g_params, decoder_LN_params, sequence_head)
        inner_loss_tuple += (inner_loss_final,)

        head_embed_new = self.psi.apply({"params": psi_params}, sequence_head)
        head_embed_new = self.encoder.apply({"params": encoder_params_new}, head_embed_new)
        head_embed_new = self.h.apply({"params": h_params}, head_embed_new)

        return head_embed_new, inner_loss_tuple

      sequence = jnp.repeat(jnp.expand_dims(sequence, axis=0), repeats=self.num_heads, axis=0)

      embed_new, inner_loss_tuple = parallelize_over_heads(self.psi_params, self.phi_params,
                                                           self.encoder_params, self.g_params,
                                                           self.decoder_LN_params, self.h_params,
                                                           sequence)
      embed_new = embed_new.sum(axis=0)
      embed_new = embed_new + self.h_bias

      inner_loss_tuple_sum = ()
      for i in range(len(inner_loss_tuple)):
        inner_loss_tuple_sum += (inner_loss_tuple[i].sum(),)

      return embed_new, inner_loss_tuple_sum

    ttt_output, inner_loss_tuple = update_embed(batch)

    inner_loss_tuple_avg = ()
    for i in range(len(inner_loss_tuple)):
      inner_loss_tuple_avg += (inner_loss_tuple[i].mean(),)
    output = ttt_output

    return output, inner_loss_tuple_avg

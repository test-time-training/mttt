"""
This code is adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/classification.py
"""

from functools import partial, lru_cache

from datasets.tfds import DataSource
from datasets import input_pipeline
import pp.builder as pp_builder
from tools import utils as u

import jax
import jax.numpy as jnp
import numpy as np


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@lru_cache(None)
def get_eval_fn(predict_fn, loss_name, layer_num, itr_num):
  """Produces eval function, also applies pmap."""
  @partial(jax.pmap, axis_name='batch')
  def _eval_fn(params, batch, labels, mask, rngs_test):
    logits, inner_loss_tuple_lyr, rngs_test = predict_fn(params, batch['image'], rngs_test)

    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)

    inner_loss_tuple_layers_avg = ()
    for layer in range(layer_num):
      inner_loss_tuple_layer_avg = ()
      for itr in range(itr_num):
        inner_loss_tuple_layer_avg += (jax.lax.pmean(inner_loss_tuple_lyr[layer][itr], 'batch'),)
      inner_loss_tuple_layers_avg += (inner_loss_tuple_layer_avg,)

    losses = getattr(u, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    return ncorrect, loss, n, inner_loss_tuple_layers_avg, rngs_test

  return _eval_fn


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size, loss_name,
               cache_final=True, cache_raw=False, prefetch=1,
               label_key='labels', layer_num=None, itr_num=None, **kw):

    data = DataSource(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
      data.get_tfdata(ordered=True), pp_fn, batch_size,
      num_ex_per_process=data.num_examples_per_process(),
      cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_input_pipeline(self.ds, prefetch)
    self.eval_fn = get_eval_fn(predict_fn, loss_name, layer_num, itr_num)
    self.label_key = label_key
    self.layer_num = layer_num
    self.itr_num = itr_num

  def run(self, params, rngs_test):
    """Computes all metrics."""
    ncorrect, loss, nseen = 0, 0, 0
    inner_loss_list = [[[] for itr in range(self.itr_num)] for layer in range(self.layer_num)]

    for step, batch in zip(range(self.steps), self.data_iter):
      labels, mask = batch.pop(self.label_key), batch.pop('_mask')
      batch_ncorrect, batch_losses, batch_n, inner_loss_tuple_layers, rngs_test, \
        = self.eval_fn(params, batch, labels, mask, rngs_test)

      ncorrect += np.sum(np.array(batch_ncorrect[0]))
      loss += np.sum(np.array(batch_losses[0]))
      nseen += np.sum(np.array(batch_n[0]))

      if step < self.steps - 1:
        for layer in range(self.layer_num):
          for itr in range(self.itr_num):
            # inner_loss_tuple_layers[layer][itr] contains for all devices the same
            inner_loss_list[layer][itr].append(inner_loss_tuple_layers[layer][itr][0])

    yield ('prec@1', ncorrect / nseen)
    yield ('loss', loss / nseen)
    yield ('inner_loss', inner_loss_list)
    yield ('rngs_test', rngs_test)

"""
This code is forked from
https://github.com/google-research/big_vision/blob/main/big_vision/input_pipeline.py
"""

import collections
import functools
import itertools
import math

from datasets.tfds import DataSource
import pp.builder as pp_builder
from tools import utils as u
import einops
import flax.jax_utils as flax_utils
import jax
import tensorflow as tf


def make_for_train(
    data, preprocess_fn, batch_size,
    shuffle_buffer_size, cache_raw=False, filter_fn=None,
    num_parallel_calls=100, prefetch=2):
  """Make an input pipeline for training."""

  data = _add_tpu_host_options(data)

  # Use data filtering at your own risk: the actual split sizes won't be known
  # in advance, so many things can go wrong in the code.
  if filter_fn:
    data = data.filter(filter_fn)

  data = data.cache() if cache_raw else data
  data = data.repeat(None)  # repeat data indefinitely

  data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data

  data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
  # Drop remainder makes shape fully static, so we can later use it if needed.
  if batch_size:
    data = data.batch(batch_size // jax.process_count(), drop_remainder=True)
  return data.prefetch(prefetch)


def training(input_config):
  batch_size = input_config.batch_size
  train_data = DataSource(**input_config.data)
  train_ds = make_for_train(
      data=train_data.get_tfdata(ordered=False),
      batch_size=batch_size,
      preprocess_fn=pp_builder.get_preprocess_fn(input_config.get("pp")),
      shuffle_buffer_size=input_config.get("shuffle_buffer_size"),
      cache_raw=input_config.get("cache_raw", False),
      filter_fn=input_config.get("filter_fn"),
  )
  return train_ds, train_data.total_examples

# The pipeline below is used for evals in multi-{G,T}PU and multi-host settings.
# As the total number of examples may not be evenly divisible accross all
# devices, we use the `infinite tf.data padding` trick, which was suggested by
# Andreas Steiner and also implemented by him in the clu library:
# https://github.com/google/CommonLoopUtils/blob/84b777c42dfd3fb6685537138433bfeb5241a006/clu/deterministic_data.py#L304.
def make_for_inference(
    data, preprocess_fn, batch_size, num_ex_per_process,
    cache_raw=False, cache_final=False):
  """Make an input pipeline for inference."""

  data = _add_tpu_host_options(data)
  data = data.cache() if cache_raw else data
  data = data.map(_add_mask(preprocess_fn), num_parallel_calls=100)
  data = data.concatenate(_get_pad_data(data))

  local_batch_size = batch_size // jax.process_count()
  # This is just like `batch`, but allows batching elements of different shapes
  # into a tf.RaggedTensor. Elements of the same fixed shape remain tf.Tensors.
  # Since we do 'infinite' padding it is safe to drop the remainder.
  data = data.apply(tf.data.experimental.dense_to_ragged_batch(
      batch_size=local_batch_size, drop_remainder=True))

  # We need to make sure that all hosts process all data and exactly the same
  # number of batches. Below we take max per-host num examples and use it on all
  # hosts to derive the number of batches.
  num_batches = math.ceil(max(num_ex_per_process) / local_batch_size)
  data = data.take(num_batches)

  # Note we cache data after a finite number of batches is taken.
  data = data.cache() if cache_final else data
  data = data.repeat()
  return data.prefetch(1), num_batches


def _get_pad_data(data):
  def zeros_like_spec(spec):
    # For unknown/flexible dimensions (None), just use 0 instead.
    return tf.zeros([x or 0 for x in spec.shape], spec.dtype)

  zero = jax.tree_map(zeros_like_spec, data.element_spec)
  return tf.data.Dataset.from_tensors(zero).repeat()


def _add_mask(pp_fn):
  def _pp_fn(example):
    return {"_mask": tf.constant(1), **pp_fn(example)}
  return _pp_fn


def _add_tpu_host_options(data):
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  options.threading.max_intra_op_parallelism = 1
  return data.with_options(options)


def prefetch_iterator(it, n):
  """Run iterator `it` ahead for `n` steps. Adapted from flax."""
  if not n:
    yield from it
    return
  queue = collections.deque()

  def enqueue(n_steps):  # Enqueues *up to* `n` elements from the iterator.
    for data in itertools.islice(it, n_steps):
      queue.append(data)

  enqueue(n)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def shard_and_put(x, shard=True, put=True):
  # pylint: disable=protected-access
  x = x._numpy()  # avoids redundant copy when converting tf tensors to numpy.
  if shard:
    x = einops.rearrange(x, "(d l) ... -> d l ...", d=jax.local_device_count())
  if shard and put:  # Only works for pmap (for now).
    x = jax.device_put_sharded(list(x), flax_utils._pmap_device_order())
  return x
  # pylint: enable=protected-access


def start_input_pipeline(data, n_prefetch=1, shard=True):
  fn = functools.partial(shard_and_put, shard=shard, put=n_prefetch)
  it = (jax.tree_util.tree_map(fn, elem) for elem in iter(data))
  return prefetch_iterator(it, n_prefetch)


def start_ragged_input_pipeline(data, n_prefetch=1, shard=True, ragged=None):
  def maybe_shard_and_put(name, x):
    return x if name in (ragged or {}) else shard_and_put(x, shard)

  it = (u.tree_map_with_names(maybe_shard_and_put, elem) for elem in iter(data))
  return prefetch_iterator(it, n_prefetch)

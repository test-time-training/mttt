"""
This code is forked from
https://github.com/google-research/big_vision/blob/main/big_vision/datasets/core.py
"""

import functools
import jax
import tensorflow_datasets as tfds


class DataSource:
  """Use TFDS as a data source."""

  def __init__(self, name, split, data_dir=None, skip_decode=("image",)):
    self.builder = _get_builder(name, data_dir)
    self.split = split
    # Each host is responsible for a fixed subset of data
    process_splits = tfds.even_splits(split, jax.process_count())
    self.process_split = process_splits[jax.process_index()]
    self.skip_decoders = {
        f: tfds.decode.SkipDecoding()
        for f in skip_decode
        if f in self.builder.info.features
    }

  def get_tfdata(self, ordered=False):
    return self.builder.as_dataset(
        split=self.process_split,
        shuffle_files=not ordered,
        read_config=tfds.ReadConfig(
            skip_prefetch=True,  # We prefetch after pipeline.
            try_autocache=False,  # We control this, esp. for few-shot.
            add_tfds_id=True,
        ),
        decoders=self.skip_decoders)

  @property
  def total_examples(self):
    return self.builder.info.splits[self.split].num_examples

  def num_examples_per_process(self, nprocess=None):
    splits = tfds.even_splits(self.split, nprocess or jax.process_count())
    return [self.builder.info.splits[s].num_examples for s in splits]


@functools.lru_cache(maxsize=None)
def _get_builder(dataset, data_dir):
  return tfds.builder(dataset, data_dir=data_dir, try_gcs=True)

"""Preprocessing imagenet2012 dataset into TFRecord format

This code is adapted from
https://github.com/google-research/big_vision/blob/main/big_vision/tools/download_tfds_datasets.py
"""

from absl import app
import tensorflow_datasets as tfds
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

def main(argv):
  if len(argv) > 1 and "download_tfds_datasets.py" in argv[0]:
    datasets = argv[1:]
  else:
    exit(0)
  for d in datasets:
    tfds.load(name=d, download=True, data_dir='')


if __name__ == "__main__":
  app.run(main)

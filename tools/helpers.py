import os
import jax
import numpy as np

def master_print(msg, logger=None):
  if jax.process_index() == 0:
    print(msg, flush=True)
    if logger is not None:
      logger.writelines(msg)
      logger.writelines('\n')
      logger.flush()


def master_mkdir(path):
  if jax.process_index() == 0:
    os.makedirs(path, mode=0o777, exist_ok=True)


def count_param(params, cnt_outer, cnt_inner, cnt_pos):
  for k, v in params.items():
    if isinstance(v, type(params)):
      cnt_outer, cnt_inner, cnt_pos = count_param(v, cnt_outer, cnt_inner, cnt_pos)
    else:
      if 'inner' in k:
        cnt_inner += v.size
      elif 'pos_embedding' in k:
        cnt_pos += v.size
      else:
        cnt_outer += v.size
  return cnt_outer, cnt_inner, cnt_pos


def load_stat_dict(stat_dict_pth, all_stat_dict):
  for k in all_stat_dict.keys():
    all_stat_dict[k] = stat_dict_pth[k]
  return


def collect_inner_loss(all_stat_dict, inner_loss_tuple_lyr, split):
  layer_num = len(inner_loss_tuple_lyr)
  for layer in range(layer_num):
    for itr in range(len(inner_loss_tuple_lyr[layer])):
      all_stat_dict['%s/inner_loss' % (split)][layer][itr].append(np.asarray(inner_loss_tuple_lyr[layer][itr]).mean())
  return

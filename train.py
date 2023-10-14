import importlib
import multiprocessing.pool
import warnings
import os.path as osp
import sys
import functools

warnings.filterwarnings("ignore")

from absl import app
from absl import flags
import ml_collections as mlc
from ml_collections import config_flags
from tqdm import tqdm
from time import perf_counter

import jax.numpy as jnp
import flax
import optax
import tensorflow as tf
import torch

from datasets import input_pipeline as input_pipeline
from tools import utils as u, build_optax as bv_optax, evaluate
from tools.helpers import *
from model import Model

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
jax.config.parse_flags_with_absl()


def make_init_fn(model, batch_size, config):
  @functools.partial(jax.jit, backend="cpu")
  def init_fn(rng):
    bs = batch_size // jax.device_count()
    image_size = (224, 224, 3)
    no_image = jnp.zeros((bs,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init({"params": rng, "idx": rng}, no_image))["params"]
    if "init_head_bias" in config:
      params["head"]["bias"] = jnp.full_like(params["head"]["bias"], config["init_head_bias"])

    return params

  return init_fn


def make_update_fn(model, tx, layer_num, itr_num, config):
  @functools.partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn(params, opt, rng, images, labels):
    if config.get("mixup") and config.mixup.p:
      rng, (images, labels), _ = u.mixup(rng, images, labels, **config.mixup)

    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      logits, inner_loss_tuple_layers = model.apply({"params": params}, images,
                                                    rngs={"idx": rng_model_local})
      return getattr(u, config.get("loss", "sigmoid_xent"))(logits=logits, labels=labels), inner_loss_tuple_layers

    (l, inner_loss_tuple_lyr), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, images, labels)
    l, grads = jax.lax.pmean((l, grads), axis_name="batch")
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    inner_loss_tuple_layers_avg = ()
    for layer in range(layer_num):
      inner_loss_tuple_layer_avg = ()
      for itr in range(itr_num):
        inner_loss_tuple_layer_avg += (jax.lax.pmean(inner_loss_tuple_lyr[layer][itr], "batch"),)
      inner_loss_tuple_layers_avg += (inner_loss_tuple_layer_avg,)

    return params, opt, rng, l, inner_loss_tuple_layers_avg

  return update_fn


def make_update_fn_accum(model, tx, accum_time, layer_num, itr_num, config):
  @functools.partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
  def update_fn_accum(params, opt, rng, images, labels):
    if config.get("mixup") and config.mixup.p:
      rng, (images, labels), _ = u.mixup(rng, images, labels, **config.mixup)

    images = images.reshape(accum_time, -1, *images.shape[1:])
    labels = labels.reshape(accum_time, -1, *labels.shape[1:])
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      logits, inner_loss_tuple_layers = model.apply({"params": params}, images,
                                                    rngs={"idx": rng_model_local})
      return getattr(u, config.get("loss", "sigmoid_xent"))(logits=logits, labels=labels), \
             inner_loss_tuple_layers

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)

    def accumulation(carry, input_dict):
      grad_avg = carry["grad_avg"]
      images = input_dict["images"]
      labels = input_dict["labels"]

      (l, inner_loss_tuple_lyr), grad = grad_fn(params, images, labels)
      grad_avg = jax.tree_util.tree_map(lambda g_avg, g: g_avg + g / accum_time,
                                        grad_avg, grad)
      carry_new = {
        "grad_avg": grad_avg
      }
      ret = {
        "loss": l,
        "inner_loss_tuple_lyr": inner_loss_tuple_lyr
      }
      return carry_new, ret

    grad_avg = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), params)
    carry_init = {"grad_avg": grad_avg}
    input_dict = {"images": images, "labels": labels}

    carry_new, ret = jax.lax.scan(accumulation, carry_init, input_dict, accum_time)

    grad_avg = jax.lax.pmean(carry_new["grad_avg"], "batch")
    l = jax.lax.pmean(ret["loss"].mean(), "batch")
    inner_loss_tuple_lyr = ret["inner_loss_tuple_lyr"]

    inner_loss_tuple_layers_avg = ()
    for layer in range(layer_num):
      inner_loss_tuple_layer_avg = ()
      for itr in range(itr_num):
        inner_loss_tuple_layer_avg += (jax.lax.pmean(inner_loss_tuple_lyr[layer][itr].mean(), "batch"),)
      inner_loss_tuple_layers_avg += (inner_loss_tuple_layer_avg,)

    updates, opt = tx.update(grad_avg, opt, params)
    params = optax.apply_updates(params, updates)

    return params, opt, rng, l, inner_loss_tuple_layers_avg

  return update_fn_accum


def make_predict_fn(model):
  def predict_fn(params, image, rng):
    rng, rng_idx = jax.random.split(rng, 2)
    logits, inner_loss_tuple_layers = model.apply({"params": params}, image, rngs={"idx": rng_idx})
    return logits, inner_loss_tuple_layers, rng

  return predict_fn


def main(argv):
  del argv
  config = flags.FLAGS.config
  workdir = flags.FLAGS.workdir

  tf.random.set_seed(config.tf_seed)
  rng = jax.random.PRNGKey(config.get("seed", 0))

  is_master = (jax.process_index() == 0)

  if is_master:
    master_mkdir(workdir)  # save log.txt, training statistics
    master_mkdir(osp.join(workdir, "../../ckpt", workdir.split("/")[-1]))  # save model checkpoint
    logger = open(osp.join(workdir, "log.txt"), "w")
  else:
    logger = None

  master_print(str(config), logger)

  save_ckpt_path = osp.join(workdir, "../../ckpt", workdir.split("/")[-1], "checkpoint.npz")
  save_stat_dict_path = osp.join(workdir, "all_stat_dict.pth")

  pool = multiprocessing.pool.ThreadPool()
  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image"]):
    importlib.import_module(f"pp.{m}")

  master_print("Initializing...")
  batch_size = config.input.batch_size
  accum_time = config.input.accum_time
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by device number ({jax.device_count()})")
  master_print(
    "Global batch size {} on {} hosts results in {} local batch size. With {} dev per host ({} dev total), "
    "that's a {} per-device batch size accumulated for {} steps.".format(
      batch_size, jax.process_count(), batch_size // jax.process_count(),
      jax.local_device_count(), jax.device_count(), batch_size // jax.device_count() // accum_time, accum_time)
  )

  master_print("Initializing train dataset...")
  n_prefetch = config.get("prefetch_to_device", 1)
  config.input.data.data_dir = config.tfds_path
  config.evals.data.data_dir = config.tfds_path
  train_ds, ntrain_img = input_pipeline.training(config.input)
  train_iter = input_pipeline.start_input_pipeline(train_ds, n_prefetch)
  total_steps = u.steps("total", config, ntrain_img, batch_size)
  steps_per_epoch = total_steps // config.total_epochs
  master_print("Running for {} steps, that means {} epochs, {} steps per epoch".format(
    total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch))

  master_print(f"Initializing model...")
  model_config = config.get("model", "tiny")
  if config.get("benchmark", "pixel") == "pixel":
    patch_size = (1, 1)
    posemb = "learn"
  else:
    patch_size = (16, 16)
    posemb = "sincos2d"

  if model_config == "small":
    model_config = dict(width=384,
                        depth=12,
                        mlp_dim=1536,
                        num_heads=6,
                        patch_size=patch_size,
                        posemb=posemb)
  elif model_config == "tiny":
    model_config = dict(width=192,
                        depth=12,
                        mlp_dim=768,
                        num_heads=3,
                        patch_size=patch_size,
                        posemb=posemb)
  else:
    raise NotImplementedError("Model %s not implemented" % model_config)

  layer_num = model_config["depth"]
  itr_num = config.inner.TTT.inner_itr + 1 if config.inner.layer_type == 'TTT' else 2

  model = Model(num_classes=config.num_classes,
                config=config.inner, **model_config)

  rng, rng_init = jax.random.split(rng)
  init_fn = make_init_fn(model, batch_size, config)
  params_cpu = init_fn(rng_init)

  outer_param_count, inner_param_count, pos_embed_param_count = count_param(params_cpu, 0, 0, 0)
  total_param_count = sum(x.size for x in jax.tree_util.tree_leaves(params_cpu))
  master_print("+Inner Param #: {}".format(inner_param_count), logger)
  master_print("+Outer Param #: {}".format(outer_param_count), logger)
  master_print("+Pos Embed Param #: {}".format(pos_embed_param_count), logger)
  master_print("Total Param #: {}".format(inner_param_count + outer_param_count), logger)
  master_print("Total Param # (+pos): {}".format(total_param_count), logger)

  master_print(f"Initializing {config.optax_name} optimizer...")
  schedule_list = []
  if not config.inner.TTT.train_init:
    schedule_list.append((".*/inner.*/.*", None))
  schedule_list.append((".*", dict(warmup_steps=10_000, decay_type="cosine")))

  config = config.to_dict()
  config["schedule"] = schedule_list
  config = mlc.ConfigDict(config)

  tx, sched_fns = bv_optax.make(config, params_cpu, sched_kw=dict(
    total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))
  opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)

  predict_fn = make_predict_fn(model)
  evaluator = evaluate.Evaluator(predict_fn=predict_fn, batch_size=config.input.batch_size,
                                 layer_num=layer_num, itr_num=itr_num, **config.evals)

  all_stat_dict = {}
  all_stat_dict["train/inner_loss"] = [[[] for i in range(itr_num)] for _ in range(layer_num)]
  all_stat_dict["val/inner_loss"] = [[[] for i in range(itr_num)] for _ in range(layer_num)]
  all_stat_dict["train/loss"] = []
  all_stat_dict["val/loss"] = []
  all_stat_dict["val/prec@1"] = []

  if save_ckpt_path and osp.exists(save_ckpt_path) and config.resume:
    resume_ckpt_path = save_ckpt_path
    resume_stat_dict_path = save_stat_dict_path

    master_print("Resume training from checkpoint...")
    checkpoint = {
      "params": params_cpu,
      "opt": opt_cpu,
    }
    checkpoint_tree = jax.tree_structure(checkpoint)
    loaded = u.load_checkpoint(checkpoint_tree, resume_ckpt_path)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree_map(u.recover_dtype, loaded)
    params_cpu, opt_cpu = checkpoint["params"], checkpoint["opt"]

    stat_dict_pth = torch.load(resume_stat_dict_path)
    load_stat_dict(stat_dict_pth, all_stat_dict)

  master_print("Kicking off misc stuff...")
  first_step = bv_optax.get_count(opt_cpu)

  master_print(f"Replicating...\n")
  params_repl = flax.jax_utils.replicate(params_cpu)
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  rng, rng_loop, rng_test = jax.random.split(rng, 3)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  rngs_test = flax.jax_utils.replicate(rng_test)
  master_print(f"First step compilations...\n")

  if accum_time > 1:
    update_fn = make_update_fn_accum(model, tx, accum_time, layer_num, itr_num, config)
  else:
    update_fn = make_update_fn(model, tx, layer_num, itr_num, config)

  train_start_time = perf_counter()
  step_start_time = perf_counter()
  with tqdm(total=(total_steps - first_step)) as t:
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
      if (step % steps_per_epoch == 1) and (step // steps_per_epoch < config.total_epochs):
        ep_stat_dict = {}
        ep_stat_dict["train/inner_loss"] = [[[] for i in range(itr_num)] for _ in range(layer_num)]
        ep_stat_dict["train/loss"] = []

      params_repl, opt_repl, rngs_loop, loss_value, inner_loss_tuple_layers_train \
        = update_fn(params_repl, opt_repl, rngs_loop, batch["image"], batch["labels"])

      ep_stat_dict["train/loss"].append(np.asarray(loss_value)[0])

      for layer in range(layer_num):
        for itr in range(itr_num):
          ep_stat_dict["train/inner_loss"][layer][itr].append(np.asarray(inner_loss_tuple_layers_train)[layer][itr][0])

      wall_time = perf_counter() - train_start_time
      current_step_time = perf_counter() - step_start_time
      eta = (total_steps - step) * current_step_time
      t.set_description(f"Wall Time: {u.hms(wall_time)} | ETA: {u.hms(eta)} | Total: {u.hms(wall_time + eta)}")

      # Epoch ends (last epoch has a little more data)
      if (step % steps_per_epoch == 0 and step // steps_per_epoch < config.total_epochs) or (step == total_steps):
        # Average epoch training stats
        all_stat_dict["train/loss"].append(np.asarray(ep_stat_dict["train/loss"]).mean())
        collect_inner_loss(all_stat_dict, ep_stat_dict["train/inner_loss"], "train")

        # Evaluation
        master_print(f"Val evaluation...\n")
        for key, value in evaluator.run(params_repl, rngs_test):
          if key == "rngs_test":
            rngs_test = value
          else:
            if key != "inner_loss":
              all_stat_dict[f"val/{key}"].append(value)
            else:
              collect_inner_loss(all_stat_dict, value, "val")

        # Checkpoint saving
        if (save_ckpt_path and is_master):
          params_cpu = jax.tree_map(lambda x: np.array(x[0]), params_repl)
          opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)
          ckpt = {"params": params_cpu, "opt": opt_cpu}
          ckpt_writer = pool.apply_async(u.save_checkpoint, (ckpt, save_ckpt_path, None))

          current_finished_ep = step // steps_per_epoch
          if current_finished_ep % 10 == 1 and current_finished_ep != config.total_epochs:
            ep_ckpt_path = osp.join(workdir, "../../ckpt", workdir.split("/")[-1], f"epoch_{current_finished_ep}")
            master_mkdir(ep_ckpt_path)
            ckpt_writer = pool.apply_async(u.save_checkpoint, (ckpt, osp.join(ep_ckpt_path, "checkpoint.npz"), None))
          torch.save(all_stat_dict, osp.join(workdir, "all_stat_dict.pth"))

      t.update()
      step_start_time = perf_counter()

  train_time = perf_counter() - train_start_time
  master_print(f"Overall Time: {u.hms(train_time)}", logger)
  master_print(f"Done!\n")
  pool.close()
  pool.join()
  if is_master:
    logger.close()


if __name__ == "__main__":
  if jax.process_index() != 0:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

  app.run(main)

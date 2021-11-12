"""Script used to train neural parts."""
import argparse
from collections import defaultdict

import json
import random
import os
import string
import subprocess
import sys

import numpy as np
from pyrender import primitive
from seaborn.palettes import color_palette
import torch
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader

from arguments import add_dataset_parameters
from np_utils import load_config

from neural_parts.datasets import build_dataloader
from neural_parts.losses import get_loss, get_loss_eval
from neural_parts.metrics import get_metrics
from neural_parts.models import optimizer_factory, build_network
from neural_parts.stats_logger import StatsLogger
import wandb
import dotenv
dotenv.load_dotenv()
sys.path.insert(0, '../../')
sys.path.insert(0, '.')
os.environ['WANDB_PROJECT'] = "neural_parts"

from data import imnet_shapenet_dataset
from utils import visualizer_util
import yaml
import copy

def set_num_threads(nt):
    nt = str(nt)
    os.environ["OPENBLAS_NUM_THREDS"] = nt
    os.environ["NUMEXPR_NUM_THREDS"] = nt
    os.environ["OMP_NUM_THREDS"] = nt
    os.environ["MKL_NUM_THREDS"] = nt


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def yield_infinite(iterable):
    while True:
        for item in iterable:
            yield item


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(experiment_directory, "opt_{:05d}").format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory, args):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    # The optimizer is wrapped with an object implementing gradient
    # accumulation
    torch.save(
        optimizer.optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )

def init_dataset_loader(cfg, data_list_path=None, gen=False, gen_latent=False, train=True):
    if gen or gen_latent:
        assert data_list_path is not None
    train_kwargs = cfg['data']['train']['kwargs']
    train_kwargs.update(cfg['data']['common']['kwargs'])
    train_dataloader_kwargs = cfg['data']['train'].get(
        'dataloader_kwargs', {})
    train_dataloader_kwargs.update(cfg['data']['common'].get(
        'dataloader_kwargs', {}))
    train_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
        primitive_num=cfg['model']['kwargs']['primitive_num'],
        **train_kwargs)
    val_kwargs = cfg['data']['val']['kwargs']
    val_kwargs.update(cfg['data']['common']['kwargs'])
    val_dataloader_kwargs = cfg['data']['val'].get(
        'dataloader_kwargs', {})
    val_dataloader_kwargs.update(cfg['data']['common'].get(
        'dataloader_kwargs', {}))
    val_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
        primitive_num=cfg['model']['kwargs']['primitive_num'],
        split=('train' if train else 'eval'),
        **val_kwargs)
    test_kwargs = cfg['data']['test']['kwargs']
    test_kwargs.update(cfg['data']['common']['kwargs'])
    test_dataloader_kwargs = cfg['data']['test'].get(
        'dataloader_kwargs', {})
    test_dataloader_kwargs.update(cfg['data']['common'].get(
        'dataloader_kwargs', {}))
    if gen or gen_latent:
        test_kwargs['list_path'] = data_list_path
    test_dataset = imnet_shapenet_dataset.IMNetShapeNetDataset(
        primitive_num=cfg['model']['kwargs']['primitive_num'],
        split=('train' if train else 'eval'),
        **test_kwargs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        num_workers=10,
        pin_memory=True,
        shuffle=True,
        **train_dataloader_kwargs)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['eval']['batch_size'],
        shuffle=False,
        **val_dataloader_kwargs)
    if gen or gen_latent:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **test_dataloader_kwargs)
    else:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg['training'].get(
                'test',
                {'batch_size': cfg['training']['eval']['batch_size']
                    })['batch_size'],
            shuffle=False,
            **test_dataloader_kwargs)
    vis_dataloader_kwargs = copy.deepcopy(val_dataloader_kwargs)

    vis_dataloader_kwargs.update(cfg['training']['visualize'].get(
        'dataloader_kwargs', {}))
    shuffle_seed = cfg['training']['visualize'].get(
        'dataloader_shuffle_seed', 0)
    worker_init_fn = None
    if shuffle_seed is not None:
        worker_init_fn = lambda x: np.random.seed(shuffle_seed)
    vis_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['visualize']['batch_size'],
        worker_init_fn=worker_init_fn,
        shuffle=True,
        **vis_dataloader_kwargs)

    return {
        "test_dataset": test_dataset,
        "val_dataset": val_dataset,
        "train_dataset": train_dataset,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "train_dataloader": train_dataloader,
        "vis_dataloader": vis_dataloader
    }
    # self.vis_dataloader = DataLoader(
    #     self.val_dataset,
    #     batch_size=cfg['training']['visualize']['batch_size'],
    #     shuffle=False,
    #     **val_dataloader_kwargs)

def _sample_points_equal(points, labels):
    labels = labels.clamp(max=1)
    assert points.ndim == 3
    assert labels.ndim == 2
    n_positive = torch.sum(labels, 1, keepdim=True)
    n_negative = labels.shape[1] - n_positive
    p = n_negative * labels + n_positive * (1-labels)
    p /= torch.sum(p, 1, keepdim=True)
    return (
        points,
        labels.unsqueeze(-1),
        ((1.0/points.shape[1]) / p).unsqueeze(-1)
    )

def collapse_sample(sample):
    s0 = sample['surface_points']
    s1, s2, s3 = _sample_points_equal(sample['points'], sample['values'])
    s4 = torch.cat([sample['surface_points'], sample['surface_normals']], -1)
    return [s0, s1, s2, s3, s4]

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )

    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--moving_primitive_config_path",
        type=str,
        required=True,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Seed for the PRNG"
    )
    add_dataset_parameters(parser)
    args = parser.parse_args(argv)

    ppd_cfg = yaml.safe_load(open(args.moving_primitive_config_path))
    ppd_cfg['data']['common']['kwargs']['classes'] = [args.classes]
    set_num_threads(1)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_tag))

    # Log the training stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "a"
    ))

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    config = load_config(args.config_file)
    config["network"]["n_primitives"] = ppd_cfg['model']['kwargs']['primitive_num'] 
    primitive_num = ppd_cfg['model']['kwargs']['primitive_num'] 

    all_configs = {"args:": vars(args), "config": config, "ppd_config": ppd_cfg}
    wandb.init(config=all_configs)

    # Instantiate a dataloader to generate the samples for training
    dataloader = build_dataloader(
        config,
        args.model_tags,
        args.category_tags,
        config["training"].get("splits", ["train", "val"]),
        config["training"].get("batch_size", 32),
        args.n_processes
    )
    # Instantiate a dataloader to generate the samples for validation
    val_dataloader = build_dataloader(
        config,
        args.model_tags,
        args.category_tags,
        config["validation"].get("splits", ["test"]),
        config["validation"].get("batch_size", 8),
        args.n_processes,
        random_subset=args.val_random_subset,
        shuffle=False
    )
    dataloader_ret = init_dataset_loader(ppd_cfg)

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        config, args.weight_file, device=device
    )
    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], network.parameters())
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Create the loss and metrics functions
    loss_fn = get_loss(config["loss"]["type"], config["loss"])
    loss_eval = copy.deepcopy(config)
    types = []
    for t in loss_eval['loss']['type']:
        print(t)
        if t == 'normal_consistency_loss':
            continue
        else:
            types.append(t)
    loss_fn_eval = get_loss(types, config["loss"])
    metrics_fn = get_metrics(config["metrics"])

    batch_vis = next(iter(dataloader_ret['vis_dataloader']))
    vis_sample = collapse_sample(batch_vis)
    vis_sample[0].to(device)
    for idx, yi in enumerate(vis_sample[1:]):
        vis_sample[idx+1] = yi.to(device).requires_grad_(True)

    color_palette = visualizer_util.Visualizer.get_colorpalette('hls', ppd_cfg['model']['kwargs']['primitive_num'])
    cnt = 0
    for i in range(args.continue_from_epoch, epochs):
        network.train()
        for b, sample in zip(list(range(len(dataloader_ret['train_dataloader']))), yield_infinite(dataloader_ret['train_dataloader'])):
        # for b, sample in zip(list(range(steps_per_epoch)), yield_infinite(dataloader)):
            sample = collapse_sample(sample)
            X = sample[0].to(device)
            # (Pdb) sample[0].shape
            # torch.Size([3, 3, 137, 137])
            # (Pdb) sample[1].shape
            # torch.Size([3, 5000, 3])
            # (Pdb) sample[2].shape
            # torch.Size([3, 5000, 1])
            # (Pdb) sample[3].shape
            # torch.Size([3, 5000, 1])
            # (Pdb) sample[4].shape
            # torch.Size([3, 2000, 6])
            targets = [yi.to(device).requires_grad_(True) for yi in sample[1:]]

            # Train on batch
            batch_loss = train_on_batch(
                network, optimizer, loss_fn, metrics_fn, X, targets, config
            )
            cnt += 1 
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)
            if cnt % 10 == 0:
                StatsLogger.instance().wandb_log(i+1, b+1, batch_loss, iter=cnt, prefix='train')
            # if cnt % 20 == 0:
            # if cnt % 20 == 0 and cnt > 1:
            if cnt % ppd_cfg['training']['visualize']['every'] == 0:
                network.eval()
                # X = vis_sample[0]
                # targets = vis_sample[1:]
                sample = collapse_sample(batch_vis)
                X = sample[0].to(device)
                targets = [
                    yi.to(device).requires_grad_(True) for yi in sample[1:]
                ]
                ps = []
                with torch.no_grad():
                    for bidx in range(X.shape[0]):
                        x = X[bidx][None]
                        ts = [t[bidx][None] for t in targets]
                        ret = validate_on_batch(
                            network, loss_fn, metrics_fn, x, ts, config, no_loss=True
                        )
                        ps.append(ret['predictions']["y_prim"])

                figs = []
                gt_figs = []
                for bidx in range(X.shape[0]):
                    point_set = []
                    gt_point_set = []
                    for pidx in range(primitive_num):
                        points = ps[bidx][0, :, pidx, :].detach().cpu().numpy()
                        point_set.append(points)
                        gt_points = batch_vis["surface_points"][bidx].detach().cpu().numpy()
                        gt_point_set.append(gt_points)
                    plot = visualizer_util.get_scatter_gofig(point_set, colors=color_palette)
                    fig = visualizer_util.gen_image_from_plot(plot)
                    figs.append(fig)
                    gt_plot = visualizer_util.get_scatter_gofig(gt_point_set, colors=color_palette)
                    gt_fig = visualizer_util.gen_image_from_plot(gt_plot)
                    gt_figs.append(gt_fig)
                wandb.log({"recon": [wandb.Image(image) for image in figs], "gt": [wandb.Image(image) for image in gt_figs]}, step=cnt)
                network.train()

            if cnt % ppd_cfg['training']['checkpoint']['every'] == 0:
                save_checkpoints(
                    cnt,
                    network,
                    optimizer,
                    experiment_directory,
                    args
                )

            # if cnt % 20 == 0:
            if cnt % ppd_cfg['training']['eval']['every'] == 0:
            # if cnt % val_every == 0 and i > 0:
                StatsLogger.instance().clear()
                print("====> Validation Epoch ====>")
                network.eval()
                losses = defaultdict(lambda: 0.)
                val_cnt = 0.
                for b, sample in zip(range(len(dataloader_ret['val_dataloader'])), dataloader_ret['val_dataloader']):
                    sample = collapse_sample(sample)
                    X = sample[0].to(device)
                    targets = [
                        yi.to(device).requires_grad_(True) for yi in sample[1:]
                    ]

                    # Validate on batch
                    with torch.no_grad():
                        ret = validate_on_batch(
                            network, loss_fn_eval, metrics_fn, X, targets, config
                        )
                    batch_loss = ret['loss']
                    for n, k in StatsLogger.instance()._values.items():
                        losses[n] += k.value

                    # Print the training progress
                    StatsLogger.instance().print_progress(1, b+1, batch_loss)
                    val_cnt += 1

                for n in losses:
                    losses[n] /= val_cnt
                StatsLogger.instance().wandb_log(1, b+1, batch_loss, iter=cnt, losses=losses, prefix='val')
                StatsLogger.instance().clear()
                print("====> Validation Epoch ====>")
                network.train()
        StatsLogger.instance().clear()

    print("Saved statistics in {}".format(experiment_tag))


if __name__ == "__main__":
    main(sys.argv[1:])

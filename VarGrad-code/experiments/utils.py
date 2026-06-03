import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods import METHODS

DYNAMIC_PSMGD_DEFAULT_THRESHOLDS = {
    "refresh_rel_fro": {
        "below": 1.0164316892623901,
        "above": 1.098314642906189,
    },
    "step_rel_fro": {
        "below": 1.1187902688980103,
        "above": 1.7600570917129517,
    },
}
DYNAMIC_PSMGD_DEFAULT_DIRECTIONS = {
    "refresh_rel_fro": "below",
    "step_rel_fro": "above",
}


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument("--n-epochs", type=int, default=300)
common_parser.add_argument("--batch-size", type=int, default=120, help="batch size")
common_parser.add_argument(
    "--method", type=str, choices=list(METHODS.keys()), help="MTL weight method"
)
common_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
common_parser.add_argument(
    "--method-params-lr",
    type=float,
    default=0.025,
    help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")
common_parser.add_argument("--save-dir", type=str, default="./save", help="path to save experiment stats")
# NashMTL
common_parser.add_argument(
    "--nashmtl-optim-niter", type=int, default=20, help="number of CCCP iterations"
)
common_parser.add_argument(
    "--update-weights-every",
    type=int,
    default=1,
    help="update task weights every x iterations.",
)
# stl
common_parser.add_argument(
    "--main-task",
    type=int,
    default=0,
    help="main task for stl. Ignored if method != stl",
)
# cagrad
common_parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
# fairgrad
common_parser.add_argument("--alpha", type=float, default=1.0, help="alpha for FairGrad alg.")
common_parser.add_argument(
    "--preprocessing",
    type=str,
    choices=["identity", "vargrad"],
    default="vargrad",
    help="gradient preprocessing module",
)
common_parser.add_argument(
    "--solver",
    type=str,
    choices=["fairgrad"],
    default="fairgrad",
    help="baseline solver used to generate candidate task weights",
)
common_parser.add_argument(
    "--scheduler",
    type=str,
    choices=["every_step", "psmgd_periodic", "psmgd_dynamic"],
    default="every_step",
    help="weight scheduling strategy",
)
common_parser.add_argument(
    "--use-vargrad",
    type=str2bool,
    default=None,
    help="optional override for preprocessing module",
)
common_parser.add_argument(
    "--use-psmgd",
    type=str2bool,
    default=None,
    help="optional override for scheduler",
)
common_parser.add_argument(
    "--psmgd-R",
    type=int,
    default=10,
    help="period length for PSMGD-style updates",
)
common_parser.add_argument(
    "--psmgd-alpha",
    type=float,
    default=0.5,
    help="EMA smoothing coefficient for periodic weight refreshes",
)
common_parser.add_argument(
    "--psmgd-dynamic-threshold",
    type=float,
    default=None,
    help="selected metric threshold for dynamic PSMGD solver refreshes",
)
common_parser.add_argument(
    "--psmgd-dynamic-metric",
    type=str,
    choices=["refresh_rel_fro", "step_rel_fro"],
    default="refresh_rel_fro",
    help="metric used by dynamic PSMGD solver refreshes",
)
common_parser.add_argument(
    "--psmgd-dynamic-direction",
    type=str,
    choices=["above", "below"],
    default=None,
    help="comparison direction used by dynamic PSMGD solver refreshes",
)
common_parser.add_argument(
    "--log-solver-updates",
    type=str2bool,
    default=True,
    help="log steps where the solver is called and task weights are refreshed",
)
common_parser.add_argument(
    "--save-u-telemetry",
    type=str2bool,
    default=False,
    help="save lightweight U_t telemetry to a JSONL file",
)
# famo
common_parser.add_argument("--gamma", type=float, default=0.01, help="gamma of famo")
common_parser.add_argument("--use_log", action='store_true', help="whether use log for famo")
common_parser.add_argument("--max_norm", type=float, default=1.0, help="beta for RMS_weight alg.")
common_parser.add_argument("--task", type=int, default=0, help="train single task number for (celeba)")
# dwa
common_parser.add_argument(
    "--dwa-temp",
    type=float,
    default=2.0,
    help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.",
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def _to_serializable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def log_solver_update_event(extra_outputs, epoch, batch_idx, global_step, enabled=True):
    if not enabled or not extra_outputs:
        return
    if not extra_outputs.get("solver_called", False):
        return
    if not extra_outputs.get("updated_weights", False):
        return

    logging.info(
        "[solver_update] global_step=%s scheduler_step=%s epoch=%s batch=%s "
        "scheduler=%s preprocessing=%s solver=%s dynamic_score=%s threshold=%s",
        global_step,
        extra_outputs.get("scheduler_step"),
        epoch,
        batch_idx,
        extra_outputs.get("scheduler"),
        extra_outputs.get("preprocessing"),
        extra_outputs.get("solver"),
        extra_outputs.get("dynamic_refresh_score"),
        extra_outputs.get("dynamic_refresh_threshold"),
    )


def write_u_telemetry_event(file_obj, extra_outputs, global_step):
    if file_obj is None or not extra_outputs:
        return
    event = extra_outputs.get("u_telemetry")
    if not event:
        return
    row = {"global_step": int(global_step)}
    row.update(event)
    file_obj.write(json.dumps(row) + "\n")


def resolve_fairgrad_config_from_args(args):
    preprocessing = args.preprocessing
    if args.use_vargrad is True:
        preprocessing = "vargrad"
    elif args.use_vargrad is False:
        preprocessing = "identity"

    scheduler = args.scheduler
    if args.use_psmgd is True:
        scheduler = "psmgd_periodic"
    elif args.use_psmgd is False:
        scheduler = "every_step"

    return dict(
        preprocessing=preprocessing,
        solver=args.solver,
        scheduler=scheduler,
    )


def resolve_psmgd_dynamic_direction_from_args(args):
    if args.psmgd_dynamic_direction is not None:
        return args.psmgd_dynamic_direction
    return DYNAMIC_PSMGD_DEFAULT_DIRECTIONS[args.psmgd_dynamic_metric]


def resolve_psmgd_dynamic_threshold_from_args(args):
    if args.psmgd_dynamic_threshold is not None:
        return args.psmgd_dynamic_threshold
    direction = resolve_psmgd_dynamic_direction_from_args(args)
    return DYNAMIC_PSMGD_DEFAULT_THRESHOLDS[args.psmgd_dynamic_metric][direction]


def build_experiment_output_stem(args):
    if args.method == "fairgrad":
        config = resolve_fairgrad_config_from_args(args)
        beta = getattr(args, "beta", 0.85)
        if config["scheduler"] == "psmgd_periodic":
            return (
                f"vargrad_reimpl_{config['preprocessing']}_beta{beta}"
                f"_fairgrad_alpha{args.alpha}_psmgd_R{args.psmgd_R}"
                f"_a{args.psmgd_alpha}_sd{args.seed}"
            )
        if config["scheduler"] == "psmgd_dynamic":
            direction = resolve_psmgd_dynamic_direction_from_args(args)
            threshold = resolve_psmgd_dynamic_threshold_from_args(args)
            return (
                f"vargrad_reimpl_{config['preprocessing']}_beta{beta}"
                f"_fairgrad_alpha{args.alpha}_psmgd_dynamic"
                f"_{args.psmgd_dynamic_metric}_{direction}_thr{threshold}"
                f"_a{args.psmgd_alpha}_sd{args.seed}"
            )
        return (
            f"vargrad_reimpl_{config['preprocessing']}_beta{beta}"
            f"_fairgrad_alpha{args.alpha}_{config['scheduler']}_sd{args.seed}"
        )

    if "fairgrad" in args.method:
        return f"{args.method}_alpha{args.alpha}_sd{args.seed}"
    return f"{args.method}_sd{args.seed}"


def extract_weight_method_parameters_from_args(args):
    fairgrad_config = resolve_fairgrad_config_from_args(args)
    psmgd_dynamic_direction = resolve_psmgd_dynamic_direction_from_args(args)
    psmgd_dynamic_threshold = resolve_psmgd_dynamic_threshold_from_args(args)
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
                max_norm=args.max_norm,
            ),
            stl=dict(main_task=args.main_task),
            dwa=dict(temp=args.dwa_temp),
            cagrad=dict(c=args.c, max_norm=args.max_norm),
            log_cagrad=dict(c=args.c, max_norm=args.max_norm),
            famo=dict(gamma=args.gamma,
                      w_lr=args.method_params_lr,
                      max_norm=args.max_norm),
            fairgrad_original=dict(
                alpha=args.alpha,
                max_norm=args.max_norm,
            ),
            fairgrad=dict(
                alpha=args.alpha,
                max_norm=args.max_norm,
                preprocessing=fairgrad_config["preprocessing"],
                scheduler=fairgrad_config["scheduler"],
                beta=getattr(args, "beta", 0.85),
                psmgd_R=args.psmgd_R,
                psmgd_alpha=args.psmgd_alpha,
                psmgd_dynamic_threshold=psmgd_dynamic_threshold,
                psmgd_dynamic_metric=args.psmgd_dynamic_metric,
                psmgd_dynamic_direction=psmgd_dynamic_direction,
            ),
        )
    )
    return weight_methods_parameters

"""
Cityscapes VarGrad Trainer

This module implements VarGrad trainer for the Cityscapes dataset.

Author: Anonymous
License: MIT
"""

import os
import logging
import wandb
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from experiments.utils import (
    build_experiment_output_stem,
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    log_solver_update_event,
    set_logger,
    set_seed,
    str2bool,
    write_u_telemetry_event,
)

# Local imports
from data import Cityscapes
from models import SegNet, SegNetMtan
from utils import ConfMatrix, delta_fn, depth_error
from methods.weight_methods_vargrad import WeightMethods

set_logger()


def calc_loss(x_pred, x_output, task_type):
    """
    Calculate loss for different task types.
    
    Args:
        x_pred: Predicted output
        x_output: Ground truth output
        task_type: Type of task ('semantic' or 'depth')
    
    Returns:
        Calculated loss
    """
    device = x_pred.device

    # Binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # Semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # Depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


# def get_current_beta(epoch, beta_start, beta_end, beta_decay_epochs):
#     t = min(epoch, beta_decay_epochs)
#     return beta_start + (beta_end - beta_start) * (t / beta_decay_epochs)


def main(path, lr, bs, device):
    # ----
    # Nets
    # ---
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on Cityscapes."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)
    logging.info(
        "Post-step train-mode forward compatibility: %s",
        args.post_step_train_forward,
    )

    cityscapes_train_set = Cityscapes(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    cityscapes_test_set = Cityscapes(root=path.as_posix(), train=False)


    
     # train_loader = torch.utils.data.DataLoader(
    #     dataset=cityscapes_train_set, batch_size=bs, shuffle=True, num_workers=2
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     dataset=cityscapes_test_set, batch_size=bs, shuffle=False, num_workers=2
    # )
    train_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=2, device=device, **weight_methods_parameters[args.method]
    )
    # optimizer
    optimizer = torch.optim.Adam(
       [
           dict(params=model.parameters(), lr=lr),
           dict(params=weight_method.parameters(), lr=args.method_params_lr),
       ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []
    os.makedirs(args.save_dir, exist_ok=True)
    u_telemetry_file = None
    if getattr(args, "save_u_telemetry", False):
        u_telemetry_name = build_experiment_output_stem(args)
        u_telemetry_path = os.path.join(
            args.save_dir, f"{u_telemetry_name}.u_telemetry.jsonl"
        )
        u_telemetry_file = open(u_telemetry_path, "w", buffering=1)
        logging.info("Saving U_t telemetry to %s", u_telemetry_path)
    
    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)
        
        for j, batch in enumerate(train_loader):
            custom_step += 1
            
            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth = train_depth.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
                beta=args.beta,
            )
            log_solver_update_event(
                extra_outputs,
                epoch=epoch,
                batch_idx=j,
                global_step=custom_step,
                enabled=getattr(args, "log_solver_updates", True),
            )
            write_u_telemetry_event(
                u_telemetry_file,
                extra_outputs,
                global_step=custom_step,
            )

            loss_list.append(losses.detach().cpu())
            optimizer.step()

            # if args.post_step_train_forward:
            #     # Match the original Cityscapes trainer's train-mode post-step
            #     # forward, which updates BatchNorm running statistics.
            #     with torch.no_grad():
            #         train_pred = model(train_data, return_representation=False)

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = next(test_dataset)
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [7, 8, 10, 11]]
            )
            deltas[epoch] = test_delta_m

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR"
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"TEST: {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} | "
                f"{avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}"
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)

            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",

                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
            ]

            name = build_experiment_output_stem(args)
            
            torch.save({
                "delta_m": deltas,
                "keys": keys,
                "avg_cost": avg_cost,
                "losses": loss_list,
            }, os.path.join(args.save_dir, f"{name}.stats"))

    if u_telemetry_file is not None:
        u_telemetry_file.close()

    print("Final Performance: ")
    print('TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
            .format(np.mean(avg_cost[-10:, 6]), np.mean(avg_cost[-10:, 7]), np.mean(avg_cost[-10:, 8]),
                    np.mean(avg_cost[-10:, 9]), np.mean(avg_cost[-10:, 10]), np.mean(avg_cost[-10:, 11]), np.mean(deltas[-10:])))


if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes VarGrad FairGrad", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=8,
        save_dir="/root/autodl-tmp/exp_logs_save/vargrad_reimpl/cityscapes/save",
    )
    
    # Add custom arguments
    parser.add_argument(
        "--beta",
        type=float,
        default=0.85,
        help="Beta parameter for Vargrad method",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="Model type",
    )
    parser.add_argument(
        "--apply-augmentation", 
        type=str2bool, 
        default=True, 
        help="Data augmentations"
    )
    parser.add_argument(
        "--post-step-train-forward",
        type=str2bool,
        default=True,
        help=(
            "Run one no-grad train-mode forward after optimizer.step() to match "
            "the original Cityscapes trainer BatchNorm/statistics trajectory."
        ),
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default=None, 
        help="Name of Weights & Biases Project."
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None, 
        help="Name of Weights & Biases Entity."
    )
    
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device)

    if wandb.run is not None:
        wandb.finish() 

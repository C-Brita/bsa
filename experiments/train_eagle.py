import sys
sys.path.append("../../")

import argparse
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from erwin.training import fit, load_checkpoint
from erwin.models.erwin import ErwinTransformer
from erwin.experiments.datasets import EagleDataset
from erwin.experiments.wrappers import EagleModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin",
                        help="Model type (mpnn, mace, pointtransformer, pointnetpp, erwin)")
    parser.add_argument("--data-path", type=str)    
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size (tiny, small, base)")
    parser.add_argument("--dilation", type=int, default=1,
                        help="Dilation factor for the dataset")
    parser.add_argument("--num-epochs", type=int, default=200000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Batch size for training")
    parser.add_argument("--use-wandb", action="store_true", default=True,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, default=8e-4,
                        help="Learning rate")
    parser.add_argument("--val-every-iter", type=int, default=500,
                        help="Validation frequency in iterations")
    parser.add_argument("--experiment", type=str, default="eagle",
                        help="Experiment name")
    parser.add_argument("--test", type=int, default=1,
                        help="Whether to run testing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_pe", type=int, default=0)
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 16,
        "c_hidden": 16,
        "ball_sizes": [256, 256, 256, 128, 128],
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "dimensionality": 2,
        "rotate": 45,
    },
    "medium": {
        "c_in": 16,
        "c_hidden": 32,
        "ball_sizes": [256, 256, 256, 128, 128],
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "dimensionality": 2,
        "rotate": 45,
    },
    "large": {
        "c_in": 32,
        "c_hidden": 64,
        "ball_sizes": [256, 256, 256, 128, 128],
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "dimensionality": 2,
        "rotate": 45,
    },
}

model_cls = {
    "erwin": ErwinTransformer,
}


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_window_length = 400
    train_dataset = EagleDataset(args.data_path, mode="train", window_length=6)
    valid_dataset = EagleDataset(args.data_path, mode="valid", window_length=2)
    test_dataset = EagleDataset(args.data_path, mode="test", window_length=test_window_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        collate_fn=train_dataset.collate_fn,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )

    if args.model == "erwin":
        model_config = erwin_configs[args.size]

    dynamic_model = model_cls[args.model](**model_config)
    model = EagleModel(dynamic_model, train_dataset.denormalize, use_pe=args.use_pe).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    config = vars(args)
    config.update(model_config)

    if args.num_epochs > 0:
        fit(config, model, optimizer, scheduler, train_loader, valid_loader, None, 100, 300)

    if args.test:
        print("Loading best checkpoint for testing...")
        best_val_loss, best_step = load_checkpoint(model, optimizer, scheduler, config)
        print(f"Loaded checkpoint from step {best_step} with validation loss {best_val_loss:.4f}")

        with torch.no_grad():
            error_velocity = torch.zeros(test_window_length - 1)
            error_pressure = torch.zeros(test_window_length - 1)

            for batch in test_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                rmse_velocity, rmse_pressure = model.evaluation_step(batch)
                error_velocity = error_velocity + rmse_velocity
                error_pressure = error_pressure + rmse_pressure

            error_velocity = error_velocity / len(test_loader)
            error_pressure = error_pressure / len(test_loader)

        np.savetxt(
            f"results/error_velocity_{args.seed}.csv",
            error_velocity.cpu().numpy(),
            delimiter=",",
        )
        np.savetxt(
            f"results/error_pressure_{args.seed}.csv",
            error_pressure.cpu().numpy(),
            delimiter=",",
        )

        print("Testing done!")
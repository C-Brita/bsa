import sys
sys.path.append("../../")

import argparse
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from erwin.training import fit
from erwin.models.erwin import ErwinTransformer
from erwin.experiments.datasets import MDDataset
from erwin.experiments.wrappers import MDModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin", 
                        help="Model type (mpnn, pointtransformer, pointnetpp, erwin)")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size (tiny, small, base)")
    parser.add_argument("--dilation", type=int, default=1,
                        help="Dilation factor for the dataset")
    parser.add_argument("--num-epochs", type=int, default=100000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--use-wandb", type=int, default=1,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--val-every-iter", type=int, default=5000,
                        help="Validation frequency in iterations")
    parser.add_argument("--experiment", type=str, default="md",
                        help="Experiment name")
    parser.add_argument("--test", type=int, default=1,
                        help="Whether to run testing")
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 16,
        "c_hidden": 16,
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "ball_sizes": [128, 128, 128, 64, 32],
        "rotate": 45,
    },
    "medium": {
        "c_in": 16,
        "c_hidden": 32,
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "ball_sizes": [128, 128, 128, 64, 32],
        "rotate": 45,
    },
    "large": {
        "c_in": 32,
        "c_hidden": 64,
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "ball_sizes": [128, 128, 128, 64, 32],
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

    train_dataset = MDDataset(
        directory=f"{args.data_path}/polymer_train",
        split=f"{args.data_path}/splits/train.txt",
        seq_len=16,
        traj_len=10000,
    )

    valid_dataset = MDDataset(
        directory=f"{args.data_path}/polymer_train",
        split=f"{args.data_path}/splits/val.txt",
        seq_len=16,
        traj_len=10000,
    )

    test_dataset = MDDataset(
        directory=f"{args.data_path}/polymer_test",
        split=f"{args.data_path}/splits/test_class2.txt",
        seq_len=16,
        traj_len=1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

    if args.model == "erwin":
        model_config = erwin_configs[args.size]

    dynamics_model = model_cls[args.model](**model_config)
    model = MDModel(seq_len=train_dataset.seq_len, dynamics_model=dynamics_model).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    config = vars(args)
    config.update(model_config)

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 100, 300)
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
from erwin.experiments.datasets import CosmologyDataset
from erwin.experiments.wrappers import CosmologyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin",
                        help="Model type (mpnn, pointtransformer, erwin)")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size configuration")
    parser.add_argument("--num-samples", type=int, default=8192,
                        help="Number of samples for training")
    parser.add_argument("--num-epochs", type=int, default=3000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--use-wandb", action="store_true", default=True,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--val-every-iter", type=int, default=500,
                        help="Validation frequency in iterations")
    parser.add_argument("--experiment", type=str, default="glx_node",
                        help="Experiment name")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Whether to run testing")
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 32,
        "c_hidden": 32,
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "ball_sizes": [256, 256, 256, 256],
    },
    "medium": {
        "c_in": 64,
        "c_hidden": 64,
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "ball_sizes": [256, 256, 256, 256],
    },
    "large": {
        "c_in": 128,
        "c_hidden": 128,
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "ball_sizes": [256, 256, 256, 256],
    },
}

model_cls = {
    "erwin": ErwinTransformer,
}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.model == "erwin":
        model_config = erwin_configs[args.size]
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    train_dataset = CosmologyDataset(
        task='node', 
        split='train', 
        num_samples=args.num_samples, 
        tfrecords_path=args.data_path, 
        knn=10,
    )
    val_dataset = CosmologyDataset(
        task='node', 
        split='val', 
        num_samples=512, 
        tfrecords_path=args.data_path, 
        knn=10,
    )
    test_dataset = CosmologyDataset(
        task='node', 
        split='test', 
        num_samples=512, 
        tfrecords_path=args.data_path, 
        knn=10,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    valid_loader = DataLoader(
        val_dataset,
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

    dynamic_model = model_cls[args.model](**model_config)
    model = CosmologyModel(dynamic_model).cuda()
    # model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    config = vars(args)
    config.update(model_config)

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 100, 200)
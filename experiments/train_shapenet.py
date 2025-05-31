import sys
sys.path.append("../../")

from dataclasses import asdict

import argparse
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bsa.training import fit
from bsa.models.erwin import ErwinTransformer
from bsa.models.transformer import Transformer, TransformerConfig
from bsa.experiments.datasets import ShapenetCarDataset
from bsa.experiments.wrappers import ShapenetCarModel



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin", 
                        choices=('mpnn', 'pointtransformer', 'pointnetpp', 'erwin', 'flat_transformer'))
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--attention_type", type=str, default="global",
                        choices=('global', 'ball', 'lucidrains_sparse', 'sparse'))
    parser.add_argument("--size", type=str, default="small", 
                        choices=('small', 'medium', 'large'))
    parser.add_argument("--num-epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-wandb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every-iter", type=int, default=100, 
                        help="Validation frequency")
    parser.add_argument("--experiment", type=str, default="shapenet", 
                        help="Experiment name in wandb")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn", type=int, default=8)
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 64,
        "c_hidden": [64, 64],
        "ball_sizes": [256, 256],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
    },
    "medium": {
        "c_in": 64,
        "c_hidden": [128, 128],
        "ball_sizes": [256, 256],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
    },
    "large": {
        "c_in": 64,
        "c_hidden": [256, 256],
        "ball_sizes": [256, 256],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
    },
}

cfg = TransformerConfig(
    c_in                     = 64,
    c_hidden                 = 64,
    depth                    = 18,
    num_heads                = 8,
    block_size               = 6,
    mlp_ratio                = 4,
    dimensionality           = 3,
    ball_size                = 256,
    should_rotate            = True,
    rotation_angle           = 45,
    attention_type           = "global",
    attention_kwargs = dict(
        compress_block_size         = 8,
        selected_blocks_number      = 4,
        use_compress_mlp           = True,
        compress_mlp_expand_factor = 2.0,
        use_group_selection        = True,
        group_selection_size       = 8,
        should_mask_blocks_in_ball = True,
        use_token_gated_attention  = True,
        use_coarse_q_attn          = True,
        use_coarse_q_importance    = True,
    ),
)

model_cls = {
    "erwin": ErwinTransformer,
    "flat_transformer": Transformer,
}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    train_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="train",
        knn=args.knn,
    )

    valid_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )

    test_dataset = ShapenetCarDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )

    if args.model == "erwin":
        model_config = erwin_configs[args.size]
        main_model = model_cls[args.model](**model_config)
    elif args.model == "flat_transformer":
        model_config = cfg
        model_config.attention_type = args.attention_type
        
        main_model = Transformer(model_config)
        model_config = asdict(model_config)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")
    
    model = ShapenetCarModel(main_model).cuda()
    model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=5e-5)

    config = vars(args)
    config.update(model_config)

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 110, 160)
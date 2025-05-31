import sys
import csv

sys.path.append("../../")

import torch

torch.set_float32_matmul_precision("high")

from erwinxnsa.models.transformer import Transformer, TransformerConfig
from erwinxnsa.experiments.wrappers import ShapenetCarModel
import numpy as np


def profile_model(model, input_size, batch_size=2, warmup=20, iterations=200):

    try:
        inputs = torch.randn(batch_size * input_size, 3).cuda()
        batch_idx = torch.arange(batch_size).repeat_interleave(input_size).cuda()

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(node_positions=inputs, batch_idx=batch_idx)

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []

            for _ in range(iterations):
                start.record()
                _ = model(inputs, batch_idx=batch_idx)
                end.record()
                torch.cuda.synchronize() 
                elapsed_time_ms = start.elapsed_time(end)
                times.append(elapsed_time_ms)

            print(
                f"INPUT SIZE {input_size}: AVG INFERENCE TIME = {np.mean(times):.3f} ms, STD: {np.std(times):.3f} ms"
            )

            return np.mean(times), np.std(times)
    except:
        print(f"INPUT SIZE {input_size}: ERROR")
        return None, None

input_sizes = [3586, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
results_mean = {}
results_std = {}

configs = {
    "full_attn": {
        "attention_type": "global",
        "attention_kwargs": None,
    },
    "sparse_attn": {
        "attention_type": "sparse",
        "attention_kwargs": dict(
            use_compress_mlp=True, use_coarse_q_attn=True, use_coarse_q_importance=True
        ),
    },
    "no_compress_mlp": {
        "attention_type": "sparse",
        "attention_kwargs": dict(
            use_compress_mlp=False, use_coarse_q_attn=True, use_coarse_q_importance=True
        ),
    },
    "no_coarse_q_attn": {
        "attention_type": "sparse",
        "attention_kwargs": dict(
            use_compress_mlp=True, use_coarse_q_attn=False, use_coarse_q_importance=True
        ),
    },
    "no_coarse_q_attn_no_coarse_q_importance": {
        "attention_type": "sparse",
        "attention_kwargs": dict(
            use_compress_mlp=True,
            use_coarse_q_attn=False,
            use_coarse_q_importance=False,
        ),
    },
    "ball_attn": {
        "attention_type": "ball",
        "attention_kwargs": dict(
            ball_size=256,
        ),
    },
    "no_group_sel": {
        "attention_type": "sparse",
        "attention_kwargs": dict(
            use_group_selection=False,
            use_coarse_q_attn=False,
            use_coarse_q_importance=False,
        ),
    },
    "no_group_sel_no_rot": {
        "attention_type": "sparse",
        "attention_kwargs": dict(
            use_group_selection=False,
            use_coarse_q_attn=False,
            use_coarse_q_importance=False,
        ),
        "should_rotate": False,
    },
    "no_rot": {
        "attention_type": "sparse",
        "attention_kwargs": None,
        "should_rotate": False,
    },
}


def profile_model_config(config_name, config):
    cfg = TransformerConfig(
        c_in=64,
        c_hidden=64,
        depth=18,
        num_heads=8,
        block_size=6,
        mlp_ratio=4,
        dimensionality=3,
        ball_size=256,
        should_rotate=config.get("should_rotate", True),
        rotation_angle=45,
        attention_type=config["attention_type"],
        attention_kwargs=config["attention_kwargs"],
    )
    print(f"Profiling {config_name}")
    main_model = Transformer(cfg)
    model = ShapenetCarModel(main_model).cuda()
    torch.compile(model)
    model.eval()
    means = []
    stds = []
    for input_size in input_sizes:
        mean, std = profile_model(model, input_size)
        means.append(mean)
        stds.append(std)
    return means, stds

for config_name, config in configs.items():
    means, stds = profile_model_config(config_name, config)
    results_mean[config_name] = means
    results_std[config_name] = stds

with open("results_mean.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input Size"] + list(results_mean.keys()))
    for i in range(len(input_sizes)):
        row = [input_sizes[i]] + [results_mean[key][i] for key in results_mean.keys()]
        writer.writerow(row)

with open("results_std.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input Size"] + list(results_std.keys()))
    for i in range(len(input_sizes)):
        row = [input_sizes[i]] + [results_std[key][i] for key in results_std.keys()]
        writer.writerow(row)

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Precomputes T5 text embeddings for VLABench task instructions.

Usage:
    uv run --python 3.10 -m cosmos_policy.datasets.save_vlabench_t5_text_embeddings \
        --data_dir /path/to/VLABench_release/primitive
"""

import argparse
from cosmos_policy.datasets.vlabench_dataset import VLABenchDataset
from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings for VLABench task instructions"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing VLABench task subdirectories",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    print(f"Loading VLABench dataset from {data_dir}...")
    dataset = VLABenchDataset(
        data_dir=data_dir,
        is_train=True,
        train_val_split=1.0,  # Load all data to get all unique commands
    )

    print(f"Found {len(dataset.unique_commands)} unique task instructions")
    print("Generating T5 embeddings...")
    t5_text_embeddings = generate_t5_embeddings(dataset.unique_commands)
    
    print(f"Saving embeddings to {data_dir}/t5_embeddings.pkl...")
    save_embeddings(t5_text_embeddings, data_dir)
    print("Done!")


if __name__ == "__main__":
    main()

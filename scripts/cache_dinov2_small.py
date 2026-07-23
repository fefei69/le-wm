#!/usr/bin/env python3
"""Populate or validate the shared Hugging Face cache for DINOv2-small."""

import argparse

from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="fail instead of downloading when the model is not cached",
    )
    args = parser.parse_args()
    model = AutoModel.from_pretrained(
        "facebook/dinov2-small", local_files_only=args.local_only
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"DINOv2-small cache ready ({parameters:,} frozen parameters)")


if __name__ == "__main__":
    main()

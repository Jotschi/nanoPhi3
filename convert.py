import argparse
import os
import torch
from safetensors.torch import save_model
from model import GPTConfig, GPT


def convert_single(out_dir: str):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    sf_name = "model.safetensors"

    # Load PT
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Write ST
    sf_filename = os.path.join(out_dir, sf_name)
    save_model(model, sf_filename)


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights to `safetensors` format.
    It is PyTorch exclusive for now.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to the output directory`",
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="Ignore safety prompt",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    if args.y:
        txt = "y"
    else:
        txt = input(
            "This conversion script will unpickle a pickled file, which is inherently unsafe."
            " Continue [Y/n] ?"
        )
    if txt.lower() in {"", "y"}:
        convert_single(out_dir)
    else:
        print(f"Answer was `{txt}` aborting.")

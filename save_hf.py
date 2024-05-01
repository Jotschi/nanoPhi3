import argparse
import os
import torch
from safetensors.torch import save_model
from model import GPTConfig, GPT


def convert_single(out_dir: str):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    sf_name = "model.safetensors"

    # Load PT
    #checkpoint = torch.load(ckpt_path, map_location="cpu")
    #gptconf = GPTConfig(**checkpoint['model_args'])
    #model = GPT(gptconf)
    hf_model = GPT.from_pretrained("gpt2", dict(dropout=0.0))
    print(hf_model)
    # Write ST
    #sf_filename = os.path.join(out_dir, sf_name)
    #save_model(model, sf_filename)


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights to huggingface format.
    It is PyTorch exclusive for now.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to the output directory`",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    convert_single(out_dir)

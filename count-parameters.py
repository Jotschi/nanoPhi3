"""
Count the parameter count for a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
pytorch_total_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameter: " + str(pytorch_total_params))


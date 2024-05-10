# Phi-3 Mini Notes

## Install deps

```bash

# Don't use pyhton 3.12 (Dynamo, torch compile not yet supported)
apt-get install python3.12-dev nvidia-cuda-dev
```

## Setup pyhton env

```bash
python3.11 -m venv venv
. venv/bin/activate
pip install torch transformers
pip install wheel
pip install datasets
pip install -U flash-attn --no-build-isolation
pip install wandb
```


## Prepare Dataset

```bash
python data/kleiner_astronaut/prepare_phi3.py
```

## Train

```bash
export CUDA_LAUNCH_BLOCKING=1 
export CUDA_VISIBLE_DEVICES=0
python phi3/train_phi3.py config/train_kleiner_astronaut_phi3.py
```
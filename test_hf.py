import argparse
import os
import sys
import torch
from safetensors.torch import save_model
from model import GPTConfig, GPT
from transformers import GPT2LMHeadModel, AutoTokenizer, GenerationConfig, GPT2Config
from transformers  import GPT2TokenizerFast

def test(model_path: str):
    #sf_name = "model.safetensors"
    # Load PT
    #checkpoint = torch.load(ckpt_path, map_location="cpu")
    #gptconf = GPTConfig(**checkpoint['model_args'])
    #GPT.from_pretrained("hf_out")
    #model = GPT(gptconf)
    #conf =  GPT2Config(n_embd=780, vocab_size=50304, resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2)
    #model = GPT2LMHeadModel.from_pretrained(model_path, config=conf, local_files_only=True, ignore_mismatched_sizes=True)
    
    ckpt_path = os.path.join(model_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    print(model)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "Es war einmal"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate completion
    output = model.generate(input_ids, max_length = 1000)

    # Decode the completion
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)
    
    #print(model_hf)
    # Write ST
    #sf_filename = os.path.join(out_dir, sf_name)
    #save_model(model, sf_filename)


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to test the huggingface weights.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model directory`",
    )
    args = parser.parse_args()
    model_path = args.model_path
    test(model_path)

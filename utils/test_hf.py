import argparse
import os
import sys
import torch
from safetensors.torch import save_model
from ..phi3.modeling_phi3 import Phi3Config, Phi3Model
from transformers  import AutoTokenizer

def test(model_path: str):
    #sf_name = "model.safetensors"
    # Load PT
    #checkpoint = torch.load(ckpt_path, map_location="cpu")
    #phi3conf = Phi3Config(**checkpoint['model_args'])
    #Phi3Model.from_pretrained("hf_out")
    #model = Phi3Model(phi3conf)
    #conf =  Phi3Config(n_embd=780, vocab_size=50304, resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2)
    #model = Phi3Model.from_pretrained(model_path, config=conf, local_files_only=True, ignore_mismatched_sizes=True)
    
    ckpt_path = os.path.join(model_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    phi3conf = Phi3Config(**checkpoint['model_args'])
    model = Phi3Model(phi3conf)
    
    print(model)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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

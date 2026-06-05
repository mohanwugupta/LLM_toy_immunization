from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_base_model(model_name_or_path: str = "meta-llama/Llama-3.2-1B", device: str = "auto", torch_dtype=torch.bfloat16):
    """
    Loads the Hugging Face base model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device
    )
    
    return model, tokenizer

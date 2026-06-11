import os
import pytest
import torch
import tempfile
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from src.train.train_condition import train as train_condition
from src.train.train_c4 import train_c4
from src.attack.run_attack import run_attack

class DummyArgs:
    pass

def test_end_to_end_tiny_pipeline():
    """Smoke test to ensure the training and attack pipelines can run without error on a tiny model."""
    try:
        import peft  # noqa: F401
    except Exception as exc:
        pytest.skip(f"PEFT/Transformers stack unavailable for tiny smoke test: {exc}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a tiny dummy model
        model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        
        # We need the tokenizer to have numbers, so we might just use the Llama tokenizer
        # But for speed, a tiny model with its own tokenizer is best. 
        # If the tokenizer lacks number tokens, our dataset code might map to UNK, which is fine for a smoke test.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # Fallback to standard tokenization for smoke test if needed
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        
        model_path = os.path.join(tmpdir, "tiny_model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # 2. Run C3 training
        args_c3 = DummyArgs()
        args_c3.model_name_or_path = model_path
        args_c3.condition = "C3"
        args_c3.output_dir = os.path.join(tmpdir, "adapters")
        args_c3.batch_size = 2
        args_c3.learning_rate = 1e-4
        args_c3.epochs = 1
        args_c3.context_len = 16
        args_c3.lora_r = 4
        args_c3.lora_alpha = 8
        args_c3.max_steps = 1
        
        train_condition(args_c3)
        assert os.path.exists(os.path.join(args_c3.output_dir, "C3"))
        
        # 3. Run C4 training
        args_c4 = DummyArgs()
        args_c4.model_name_or_path = model_path
        args_c4.output_dir = os.path.join(tmpdir, "adapters")
        args_c4.batch_size = 2
        args_c4.learning_rate = 1e-4
        args_c4.epochs = 1
        args_c4.context_len = 16
        args_c4.lora_r = 4
        args_c4.lora_alpha = 8
        args_c4.target_layer = -1
        args_c4.lambda_rep = 0.1
        args_c4.max_steps = 1
        
        train_c4(args_c4)
        assert os.path.exists(os.path.join(args_c4.output_dir, "C4"))
        
        # 4. Run Attack on C3
        args_attack = DummyArgs()
        args_attack.model_name_or_path = model_path
        args_attack.adapter_path = os.path.join(tmpdir, "adapters", "C3")
        args_attack.output_dir = os.path.join(tmpdir, "attack", "C3")
        args_attack.batch_size = 2
        args_attack.learning_rate = 1e-4
        args_attack.context_len = 16
        args_attack.max_steps = 10
        args_attack.attack_type = "sampled"
        args_attack.condition = "C3"
        args_attack.seed = 42
        args_attack.difficulty_level = 3
        
        run_attack(args_attack)
        # Check if the first attack step checkpoint was saved
        assert os.path.exists(os.path.join(args_attack.output_dir, "attack_step_10"))

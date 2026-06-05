from peft import LoraConfig, get_peft_model, TaskType
from transformers import PreTrainedModel

def setup_lora(model: PreTrainedModel) -> PreTrainedModel:
    """
    Wrap a Hugging Face model with LoRA using the default PEFT configuration.
    Target modules: q_proj, k_proj, v_proj, o_proj
    """
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model

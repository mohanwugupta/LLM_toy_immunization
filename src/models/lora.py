from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = object


def prepare_lora_model(
    model: PreTrainedModel,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> PreTrainedModel:
    """
    Wrap a Hugging Face model with LoRA using PEFT.
    Target modules default to attention projections: q_proj, k_proj, v_proj, o_proj.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


# Backward-compatible alias
def setup_lora(model: PreTrainedModel) -> PreTrainedModel:
    """Default LoRA setup with fixed hyperparameters."""
    return prepare_lora_model(model)

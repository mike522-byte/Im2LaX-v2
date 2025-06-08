import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from config import Config

config = Config()

def initialize_model_and_processor():
    """Initialize the Qwen2.5-VL model, tokenizer, and processor for training."""
    print(f"Loading model for training: {config.model_name}")
    
    model, processor = _load_model_and_processor(
        model_path=config.model_name,
        for_training=True
    )
    
    # Enable gradient checkpointing for training to save memory
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, processor

def load_finetuned_model(model_path):
    """Load a fine-tuned model for evaluation with KV cache enabled."""
    print(f"Loading fine-tuned model for inference: {model_path}")
    
    model, processor = _load_model_and_processor(
        model_path=model_path,
        for_training=False
    )
    
    model.eval()
    return model, processor

def _load_model_and_processor(model_path, for_training=True):
    """Internal function to load model and processor with appropriate settings."""
    
    # Load model with different settings for training vs inference
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=config.quantization_config,
        device_map=config.device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16 if config.mixed_precision == "fp16" else torch.float32,
        attn_implementation="flash_attention_2" if config.flash_attention else "eager",
        use_cache=False if for_training else True,  # Disable cache for training, enable for inference
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    return model, processor
import os
from transformers import BitsAndBytesConfig

# BitsAndBytes configuration for quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

class Config:
    # Model configuration
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "./qwen-latex-ocr-finetuned"
    final_output_dir = os.path.join(output_dir, "final")
    eval_result_dir = "evaluation_results_5.json"
    
    # Dataset configuration
    dataset_name = "linxy/LaTeX_OCR"
    dataset_config = "full"  # Options: "full", "synthetic_handwrite", "human_handwrite"
    
    # Training configuration
    batch_size = 1
    eval_examples = 1000
    gradient_accumulation_steps = 16
    num_train_epochs = 2
    learning_rate = 1e-5
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.1
    max_seq_length = 2048
    max_new_tokens = 1024
    max_steps = -1  # -1 means full epochs
    weight_decay = 0.01
    
    # Parameter-efficient fine-tuning configuration
    use_peft = False
    
    # Hardware and training precision configuration
    device_map = "auto"
    mixed_precision = "bf16"  # Options: "no", "fp16", "bf16"
    gradient_checkpointing = True
    flash_attention = False
    quantization_config = None
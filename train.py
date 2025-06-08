import torch
from trl import SFTTrainer, SFTConfig
from functools import partial

from config import Config
from data_preprocessing import (
    load_and_prepare_dataset, 
    LatexOCRDataset, 
    collate_fn
)
from model_utils import initialize_model_and_processor

config = Config()

def get_training_arguments():
    """Define the training arguments for SFTTrainer."""
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        report_to="tensorboard",
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=(config.mixed_precision == "fp16"),
        bf16=(config.mixed_precision == "bf16"),
        remove_unused_columns=False,
        dataloader_drop_last=True,
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        weight_decay=config.weight_decay,
        gradient_checkpointing=config.gradient_checkpointing,
        save_total_limit=3,
    )
    
    return training_args

def train():
    """Main training function."""
    print("Starting training process...")
    
    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Initialize model and processor
    model, processor = initialize_model_and_processor()
    
    # Prepare datasets
    train_preprocessed = LatexOCRDataset(train_dataset, 'train')
    eval_preprocessed = LatexOCRDataset(eval_dataset, 'evaluate')
    
    # Create partial collate function with processor
    collate_fn_with_processor = partial(collate_fn, processor=processor)
    
    # Set up training arguments
    training_args = get_training_arguments()
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn_with_processor,
        train_dataset=train_preprocessed,
        eval_dataset=eval_preprocessed,
        processing_class=processor.tokenizer
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {config.final_output_dir}")
    trainer.save_model(config.final_output_dir)
    processor.save_pretrained(config.final_output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    train()
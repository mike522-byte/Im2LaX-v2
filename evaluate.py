import torch
import json
import time
from tqdm import tqdm
from evaluate import load
from functools import partial

from config import Config
from data_preprocessing import (
    load_and_prepare_dataset,
    LatexOCRDataset,
    collate_eval_fn
)
from model_utils import load_finetuned_model

config = Config()

def evaluate():
    """Evaluate the fine-tuned model."""
    print("Starting evaluation process...")
    
    # Load model and processor
    model, processor = load_finetuned_model(config.final_output_dir)
    
    # Prepare dataset
    _, eval_dataset = load_and_prepare_dataset()
    eval_dataset = eval_dataset.select(range(config.eval_examples))
    eval_preprocessed = LatexOCRDataset(eval_dataset, 'evaluate')
    
    # Create partial collate function with processor
    collate_eval_fn_with_processor = partial(collate_eval_fn, processor=processor)
    
    # Create dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        eval_preprocessed,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_eval_fn_with_processor,
    )
    
    # Load metrics
    bleu_metric = load("bleu")
    exact_match_metric = load("exact_match")
    
    # Initialize result containers
    all_predictions = []
    all_references = []
    all_inference_time = []
    
    # Inference loop
    print("Running inference...")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items() if k != "references"}
        
        inference_start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
        output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        inference_end_time = time.time()
        
        # Process output texts to extract assistant response
        for i, prediction in enumerate(output_texts):
            if "assistant" in prediction.lower():
                # Extract only the assistant's response part
                prediction = prediction.split("assistant")[-1].strip()
                output_texts[i] = prediction
        
        inference_time = inference_end_time - inference_start_time
        
        all_predictions.extend(output_texts)
        all_references.extend(batch["references"])
        all_inference_time.append(inference_time)
    
    # Calculate metrics
    print("Calculating metrics...")
    bleu_results = bleu_metric.compute(predictions=all_predictions, references=all_references)
    exact_match_results = exact_match_metric.compute(predictions=all_predictions, references=all_references)
    
    # Calculate performance metrics
    mean_inference_time = sum(all_inference_time) / len(all_inference_time)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # Compile results
    results = {
        "bleu": bleu_results,
        "exact_match": exact_match_results,
        "average_inference_time(s)": mean_inference_time,
        "max_gpu_memory_allocated(MB)": peak_memory,
    }
    
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))
    
    # Save results
    with open(config.eval_result_dir, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {config.eval_result_dir}")
    
    return results

if __name__ == "__main__":
    evaluate()
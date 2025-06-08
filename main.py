import argparse
import sys
from config import Config
from train import train
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL LaTeX OCR Fine-tuning")
    parser.add_argument(
        "--mode", 
        choices=["train", "evaluate", "both"], 
        default="both",
        help="Mode to run: train, evaluate, or both"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model for evaluation (if different from config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to evaluate"
    )
    
    args = parser.parse_args()
    config = Config()
    
    if args.mode in ["train", "both"]:
        print("=" * 50)
        print("STARTING TRAINING")
        print("=" * 50)
        try:
            train()
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed with error: {e}")
            if args.mode == "both":
                print("Skipping evaluation due to training failure.")
                sys.exit(1)
    
    if args.mode in ["evaluate", "both"]:
        print("=" * 50)
        print("STARTING EVALUATION")
        print("=" * 50)
        
        model_path = args.model_path if args.model_path else config.final_output_dir
        
        try:
            results = evaluate(
                finetuned_model_path=model_path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_examples=args.num_examples
            )
            print("Evaluation completed successfully!")
            print("\nFinal Results Summary:")
            print(f"BLEU Score: {results['bleu']['bleu']:.4f}")
            print(f"Exact Match: {results['exact_match']['exact_match']:.4f}")
            print(f"Average Inference Time: {results['average_inference_time(s)']:.4f}s")
            
        except Exception as e:
            print(f"Evaluation failed with error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
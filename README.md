# LaTeX OCR Fine-tuning

This project provides a modular implementation for fine-tuning Qwen2.5-VL models on LaTeX OCR tasks.

## Fine-tuning Results
![image](https://github.com/user-attachments/assets/418fbd7d-93cf-469f-9fc9-b1023c575803)


https://github.com/user-attachments/assets/80fa8857-6c25-4f25-8292-fa57b4881c16



## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have CUDA available if you want to use GPU acceleration.

## Configuration

All configuration settings are centralized in `config.py`. Key settings include:

- **Model Configuration**: Model name, output directories
- **Dataset Configuration**: Dataset name and config
- **Training Configuration**: Batch size, learning rate, epochs, etc.
- **Hardware Configuration**: Mixed precision, gradient checkpointing, etc.

## Usage

### Training Only
```bash
python main.py --mode train
```

### Evaluation Only
```bash
python main.py --mode evaluate --model_path ./path/to/finetuned/model
```

### Both Training and Evaluation
```bash
python main.py --mode both
```

### Custom Evaluation Parameters
```bash
python main.py --mode evaluate \
    --model_path ./qwen-latex-ocr-finetuned/final \
    --batch_size 2 \
    --max_new_tokens 512 \
    --num_examples 500
```
## Output

- **Training**: Model checkpoints saved to `./qwen-latex-ocr-finetuned/`
- **Final Model**: Saved to `./qwen-latex-ocr-finetuned/final/`
- **Evaluation Results**: Saved to `evaluation_results_5.json`
- **Logs**: TensorBoard logs in the output directory

## Metrics

The evaluation computes:
- **BLEU Score**: Measures text similarity between predicted and reference LaTeX
- **CER Score**: Measure how many steps needed to edit the predicted string into becoming the label
- **Exact Match**: Percentage of exactly matching predictions
- **Inference Time**: Average time per batch
- **Memory Usage**: Peak GPU memory allocation

## Customization

To customize the training:

1. Modify `config.py` for different hyperparameters
2. Extend `LatexOCRDataset` in `data_preprocessing.py` for different data formats
3. Adjust `collate_fn` for different input processing
4. Modify training arguments in `train.py` for different training strategies

## Notes

- The code supports both full fine-tuning and quantized training
- Gradient checkpointing is enabled by default to save memory
- The model uses chat templates for consistent input formatting
- Evaluation automatically extracts assistant responses from generated text

# LaTeX OCR Fine-tuning

This project provides a modular implementation for fine-tuning VLM models on LaTeX OCR tasks.
[Read the paper](LatexOCR.pdf)

<img width="2564" height="1748" alt="SWIN-mBART" src="https://github.com/user-attachments/assets/a665d778-5164-47d2-a393-5153ee82eaf3" />

## Figure (Im2LaX Model Architecture)

## Results

<img width="1095" height="340" alt="image" src="https://github.com/user-attachments/assets/07dba068-1666-4992-910a-5c1a854cac73" />


## Visualize Comparison of Simple Test Case

<img width="754" height="347" alt="image" src="https://github.com/user-attachments/assets/92fb2f97-265f-4a5f-8a64-6f907c4e9c3b" />


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

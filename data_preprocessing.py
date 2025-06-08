import torch
from datasets import load_dataset
from PIL import Image
from config import Config

config = Config()

class LatexOCRDataset(torch.utils.data.Dataset):
    """Dataset class for LaTeX OCR with message structure."""
    
    def __init__(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load image if needed
        if isinstance(item["image"], str):
            image = Image.open(item["image"]).convert("RGB")
        else:
            image = item["image"]
        
        # Create message structure for chat template
        if self.mode == 'train':
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this mathematical formula image to LaTeX code."}
                ]},
                {"role": "assistant", "content": item["text"]}
            ]
            
            return {
                "messages": messages,
                "images": [image],
            }
        
        elif self.mode == 'evaluate':
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this mathematical formula image to LaTeX code."}
                ]},
            ]
            
            return{
                "messages": messages,
                "images": [image],
                "references": item["text"],
            }

def load_and_prepare_dataset():
    """Load and prepare the LaTeX OCR dataset."""
    print(f"Loading dataset: {config.dataset_name}, config: {config.dataset_config}")
    dataset = load_dataset(config.dataset_name, name=config.dataset_config)
    
    # Split into train and eval if not already split
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    
    return train_dataset, eval_dataset

def collate_fn(examples, processor):
    """Collate function for training."""
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"][0] for example in examples]
    
    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch

def collate_eval_fn(examples, processor):
    """Collate function for evaluation."""
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"][0] for example in examples]
    references = [example["references"] for example in examples]
    
    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    batch["references"] = references
    
    return batch
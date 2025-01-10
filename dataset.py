import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict

IMAGES_DIR = './K-data/images'
CAPTION_FILE = './K-data/labels.txt'

class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        processor = None,
        tokenizer = None,
    ):
        """
        Dataset for image captioning
        Args:
            image_dir (str): Directory containing the images
            caption_file (str): Path to text file containing captions
            processor: Image processor for the vision model
            tokenizer: Tokenizer for the language model
        """
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        
        # Read captions and image filenames
        self.items = []
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming each line is in format: image_filename.jpg|caption
                image_name, caption = line.strip().split('|')
                image_path = os.path.join(image_dir, image_name)
                
                # Only add if image file exists
                if os.path.exists(image_path):
                    self.items.append({
                        'image_path': image_path,
                        'caption': caption
                    })
                else:
                    raise Exception(f"Image {image_path} does not exist")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        
        # Load and convert image to RGB
        image = Image.open(item['image_path']).convert('RGB')
        # Process image without adding batch dimension
        pixel_values = self.processor(image).pixel_values[0]
        
        # Process text
        text = item['caption']
        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)  # Remove batch dimension
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def custom_collator(features):
    """
    Custom collator function to properly batch the dataset items
    """
    pixel_values = torch.stack([feature["pixel_values"] for feature in features])
    labels = torch.stack([feature["labels"] for feature in features])
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def get_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=16
):
    """
    Create a DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collator
    )
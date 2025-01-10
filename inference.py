import os
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from utils import setup_model_and_tokenizer, setup_device

def load_model(model_path, device=None):
    """
    Load a saved model and move it to the specified device
    """
    if device is None:
        device = setup_device()
        
    model, processor, tokenizer = setup_model_and_tokenizer()
    
    # Load the saved model weights
    if os.path.exists(model_path):
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model = model.to(device)
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
        
    return model, processor, tokenizer

def generate_caption(
    image_path,
    model,
    processor,
    tokenizer,
    device=None,
    max_length=32,
):
    """
    Generate a caption for a single image
    """
    if device is None:
        device = setup_device()
        
    try:
        # Prepare input
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        
        # Move input to the same device as model
        pixel_values = pixel_values.to(device)
        
        print(f"Input image shape: {pixel_values.shape}")
        
        # Generate with proper attention mask
        attention_mask = torch.ones(pixel_values.shape[0], 1).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id
            )[0].cpu()
        
        caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return caption
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Example usage
    model_path = "./outputs/final_model"
    image_path = os.path.join("./K-data/images", "02.jpg")
    
    # Set up device
    device = setup_device()
    
    # Load model
    model, processor, tokenizer = load_model(model_path, device)
    
    # Generate caption
    caption = generate_caption(
        image_path,
        model,
        processor,
        tokenizer,
        device
    )
    
    if caption:
        print(f"\nGenerated caption: {caption}")

if __name__ == "__main__":
    main()
import os
from transformers import Seq2SeqTrainer

from dataset import ImageCaptioningDataset, IMAGES_DIR, CAPTION_FILE
from utils import setup_model_and_tokenizer, get_training_args, setup_device

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom compute_loss that handles the num_items_in_batch parameter
        """
        if "num_items_in_batch" in inputs:
            inputs.pop("num_items_in_batch")
            
        labels = inputs.pop("labels")
        outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def train(
    batch_size=2,
    num_epochs=100,
    max_len=32,
    fp16=True,
    output_dir="./outputs"
):
    """
    Train the image captioning model
    """
    # Set device
    device = setup_device()
    
    # Initialize model, processor, and tokenizer
    model, processor, tokenizer = setup_model_and_tokenizer(max_length=max_len)
    model = model.to(device)
    
    # Create dataset
    dataset = ImageCaptioningDataset(
        image_dir=IMAGES_DIR,
        caption_file=CAPTION_FILE,
        processor=processor,
        tokenizer=tokenizer,
    )
    
    # Get training arguments
    training_args = get_training_args(
        batch_size=batch_size,
        num_epochs=num_epochs,
        fp16=fp16,
        output_dir=output_dir
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Save the final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return model, processor, tokenizer

if __name__ == "__main__":
    train()
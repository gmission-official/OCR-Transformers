import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    ViTConfig,
    GPT2Config,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

def setup_model_and_tokenizer(max_length=32):
    """
    Initialize and configure the model, processor, and tokenizer
    """
    # Configure the encoder (ViT) and decoder (GPT2)
    config_encoder = ViTConfig()
    config_decoder = GPT2Config()
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    
    # Create the model
    model = VisionEncoderDecoderModel(config=config)
    
    # Setup processor and tokenizer
    processor = AutoProcessor.from_pretrained('facebook/deit-tiny-patch16-224', use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model parameters
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = tokenizer.vocab_size
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    return model, processor, tokenizer

def get_training_args(
    batch_size=2,
    num_epochs=100,
    fp16=True,
    output_dir="./outputs",
    run_name="image_captioning"
):
    """
    Get training arguments for the Seq2SeqTrainer
    """
    from transformers import Seq2SeqTrainingArguments
    
    return Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="no",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=16,
        output_dir=output_dir,
        logging_steps=10,
        report_to="none",
        save_steps=200,
        num_train_epochs=num_epochs,
        run_name=run_name,
        remove_unused_columns=False,
        label_names=["labels"],
        learning_rate=5e-5,
        weight_decay=0.01,
    )

def setup_device():
    """
    Set up and return the appropriate device (CPU/GPU)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
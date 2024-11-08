import os
import argparse
import time
import json
import gc
import wandb
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# allow flexible memory allocation for CUDA and enable garbage collection
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'

def token_ids_to_text(token_ids, tokenizer):
    """Decodes token ids to text."""
    text = ""
    for tid in token_ids:
        if tid == tokenizer.eos_token_id:
            break
        word = tokenizer.decode(tid, skip_special_tokens=True)
        text += word
    return text

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            
            # encode fully formatted data point
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, ind):
        return self.encoded_texts[ind]

    def __len__(self):
        return len(self.data)

def format_input(data_point):
    """Format the input data point."""
    instruction = data_point["instruction"]
    input_text = data_point.get("input", "")
    
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
    return f"### Instruction:\n{instruction}"

def load_and_split_data(data_paths, train_split=0.85, test_split=0.1):
    """Load and split data into train, validation and test sets."""
    data = []
    for path in data_paths:
        with open(path, 'r') as f:
            data.extend(json.load(f))
    
    N = len(data)
    print(f'Total data: {N}')

    train_portion = int(N * train_split)
    test_portion = int(N * test_split)
    val_portion = N - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Training set length: {len(train_data)}")
    print(f"Validation set length: {len(val_data)}")
    print(f"Test set length: {len(test_data)}")

    return train_data, val_data, test_data

def setup_lora_model(model, args):
    """Setup LoRA configuration and prepare model for training."""
    # Prepare model for k-bit training if using quantization
    if args.load_in_8bit or args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Define LoRA Config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data(
        args.data_paths,
        args.train_split,
        args.test_split
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer with optional quantization
    model_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation='eager',
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Setup LoRA if specified
    if args.use_lora:
        model = setup_lora_model(model, args)
    elif torch.cuda.is_available():
        model = model.to(device)
    
    # Create datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize wandb if enabled
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=f"./{args.output_name}",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        report_to="wandb" if not args.disable_wandb else "none",
        run_name=run.name if not args.disable_wandb else None,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train model
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    # Save model and tokenizer
    if args.save_model:
        if args.use_lora:
            model.save_pretrained(args.output_name)
        else:
            model.save_pretrained(args.output_name)
            tokenizer.save_pretrained(args.output_name)
        print(f"Model {'adapter' if args.use_lora else 'and tokenizer'} saved to {args.output_name}")
    
    # Finish wandb run if enabled
    if not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model on instruction data")
    
    # Model and data arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--data-paths", type=str, nargs="+", required=True,
                        help="Paths to the instruction data JSON files")
    parser.add_argument("--output-name", type=str, required=True,
                        help="Name for the output model directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Number of steps between evaluations")
    parser.add_argument("--save-steps", type=int, default=1000,
                        help="Number of steps between model saves")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--eval-accumulation-steps", type=int, default=10,
                        help="""number of predictions steps to accumulate the output tensors for,
                        before moving the results to the CPU""")
    
    # LoRA arguments
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout value")
    parser.add_argument("--lora-target-modules", type=str, default=None,
                        help="Comma-separated list of target modules for LoRA")
    
    # Quantization arguments
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit mode")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit mode")
    
    # Data split arguments
    parser.add_argument("--train-split", type=float, default=0.85,
                        help="Proportion of data to use for training")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Proportion of data to use for testing")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="instruction-tuning",
                        help="Weights & Biases project name")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the model after training")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.load_in_8bit and args.load_in_4bit:
        parser.error("Cannot use both 8-bit and 4-bit quantization")
    
    main(args)

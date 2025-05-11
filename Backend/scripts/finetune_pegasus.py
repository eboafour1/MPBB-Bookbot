from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

# Initialize Pegasus model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load multiple datasets for broad coverage
book_ds   = load_dataset("bookcorpus", split="train")
arxiv_ds  = load_dataset("scientific_papers", "arxiv", split="train")
xsum_ds   = load_dataset("xsum", split="train")
wikihow_ds= load_dataset("wikihow", split="all")

# Concatenate datasets into one large dataset
dataset = concatenate_datasets([book_ds, arxiv_ds, xsum_ds, wikihow_ds])

# Preprocessing function for abstractive summarization fine-tuning
def preprocess(example):
    # Use 'text' or fallback to generic 'article'
    source = example.get("text") or example.get("article", "")
    target = example.get("summary") or example.get("highlights", "")
    # Tokenize inputs
    inputs = tokenizer(source, truncation=True, max_length=1024)
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, truncation=True, max_length=256)
    inputs["labels"] = labels.input_ids
    return inputs

# Apply preprocessing
tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./pegasus_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,
    save_total_limit=2
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

# Run fine-tuning
if __name__ == "__main__":
    trainer.train()
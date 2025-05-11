from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("nlpyang/bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained("nlpyang/bert-base-uncased")

# Load multiple extractive summarization datasets
cnn_ds    = load_dataset("cnn_dailymail", "3.0.0", split="train")
xsum_ds   = load_dataset("xsum", split="train")
wikihow_ds= load_dataset("wikihow", split="all")

# Helper to map documents into sentences with labels
def to_sentence_examples(example):
    # Determine source text & key highlights
    article   = example.get("article") or example.get("text", "")
    highlights= example.get("highlights", "")
    sentences = article.split(". ")
    highlight_set = {h.strip() for h in highlights.split(". ") if h.strip()}
    out = {"sentence": [], "labels": []}
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        label = 1 if any(h in s for h in highlight_set) else 0
        out["sentence"].append(s)
        out["labels"].append(label)
    return out

# Prepare each dataset
mapped_datasets = []
for ds in [cnn_ds, xsum_ds, wikihow_ds]:
    mapped = ds.map(
        to_sentence_examples,
        batched=False,
        remove_columns=ds.column_names
    )
    mapped_datasets.append(mapped)

# Concatenate all mapped datasets
combined = concatenate_datasets(mapped_datasets)

# Tokenization for classification
def preprocess(examples):
    enc = tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    enc["labels"] = examples["labels"]
    return enc

tokenized = combined.map(preprocess, batched=True, remove_columns=combined.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bertsum_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    fp16=True,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

if __name__ == "__main__":
    trainer.train()

# scripts/convert_bertsum_ckpt.py

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# 1) Load your .ckpt file (make sure this path points to the actual .ckpt file)
ckpt_path = Path(
    "/Users/nicholasdesarno/Desktop/MPBB-Bookbot/path/epoch=3.ckpt"
)
# If you need safe loading, ensure you installed pytorch-lightning or add safe globals:
# from pytorch_lightning.utilities.parsing import AttributeDict
# torch.serialization.add_safe_globals([AttributeDict])
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt.get("state_dict", ckpt)

# 2) Create a matching HF config & model for extractive (classification) BERTSum
base = "bert-base-uncased"  # replace if you used a different base
# Specify number of labels (binary classification: 2)
config = AutoConfig.from_pretrained(base, num_labels=2)
model = AutoModelForSequenceClassification.from_config(config)

# 3) Load the weights into the classification model
# strict=False allows missing or unexpected keys (e.g., Lightning wrappers)
load_res = model.load_state_dict(state_dict, strict=False)
print("Loaded weights with result:", load_res)

# 4) Save in HF format
out_dir = Path("./models/bertsum_processed")
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out_dir)

# 5) Copy over the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base)
tokenizer.save_pretrained(out_dir)

print(f"✅ Converted extractive BERTSum checkpoint saved to {out_dir}")

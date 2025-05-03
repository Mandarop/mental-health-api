# common.py
import torch
from transformers import AutoTokenizer, AutoModel

# Shared tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased")
muril_model.eval()

# Shared embedding logic
@torch.no_grad()
def embed_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    return muril_model(**inputs).last_hidden_state[:, 0, :]

# app_anxiety.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from common import embed_text

# --- FastAPI app ---
app = FastAPI()

# --- Anxiety model definition ---
class MultiLabelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# Load model only once
model = MultiLabelNet()
model.load_state_dict(torch.load("anxiety.pth", map_location="cpu"))
model.eval()

# Labels
labels = [
    "Symptom 10", "Symptom 11", "Symptom 12",
    "Symptom 13", "Symptom 14", "Symptom 15", "Symptom 16"
]

# --- Request schema ---
class TextRequest(BaseModel):
    text: str

# --- Prediction endpoint ---
@app.post("/predict/")
def predict(req: TextRequest):
    x = embed_text(req.text)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0)
        predicted = [lab for lab, p in zip(labels, probs) if p > 0.3]
    return {"predicted_symptoms": predicted}

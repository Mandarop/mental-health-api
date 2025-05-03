# app_depression.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from common import embed_text

app = FastAPI()

class MultiLabelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# Load model once
model = MultiLabelNet()
model.load_state_dict(torch.load("depression.pth", map_location="cpu"))
model.eval()

labels = [
    "Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4",
    "Symptom 5", "Symptom 6", "Symptom 7", "Symptom 8", "Symptom 9"
]

class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(req: TextRequest):
    x = embed_text(req.text)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0)
        predicted = [lab for lab, p in zip(labels, probs) if p > 0.3]
    return {"predicted_symptoms": predicted}

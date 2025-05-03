from fastapi import FastAPI
from pydantic import BaseModel
import torch
from common import MultiLabelNet, embed_text

# ——— FastAPI setup ———
app = FastAPI()

# ——— Symptom labels for Anxiety ———
anxiety_labels = [
    "Symptom 10", "Symptom 11", "Symptom 12", "Symptom 13", 
    "Symptom 14", "Symptom 15", "Symptom 16"
]

# ——— Request schema ———
class TextRequest(BaseModel):
    text: str

# ——— Prediction logic ———
def predict_symptoms(text: str, model, labels: list, threshold: float = 0.3):
    cls_repr = embed_text(text)
    with torch.no_grad():
        logits = model(cls_repr)
        probs = torch.sigmoid(logits).squeeze(0)

    mask = (probs > threshold).tolist()
    predicted = [lab for lab, keep in zip(labels, mask) if keep]
    return predicted

# ——— API endpoint ———
@app.post("/predict/")
async def predict_anxiety(req: TextRequest):
    model = MultiLabelNet(input_dim=768, output_dim=7)
    model.load_state_dict(torch.load("anxiety.pth", map_location="cpu"))
    model.eval()

    predicted = predict_symptoms(req.text, model, anxiety_labels)

    del model
    torch.cuda.empty_cache()

    return {"predicted_symptoms": predicted}

# ——— Uvicorn (for local dev only) ———
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

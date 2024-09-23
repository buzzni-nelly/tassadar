import tessadar
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification


class RegularModel:
    def __init__(self):
        self.device = torch.device("mps")
        print(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        ).to(self.device)
        self.model.eval()

    def inference(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities.cpu()[0].tolist()


class BERTProxy(tessadar.Proxy):
    def __init__(self, model, tokenizer, device, batch_size=64):
        super().__init__(batch_size=batch_size)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    async def _inference(self, arguments):
        texts = list(arguments)
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities.cpu().tolist()


regular_model = RegularModel()
bert_proxy = BERTProxy()

app = FastAPI()


class InferenceRequest(BaseModel):
    text: str


@app.post("/proxy_inference")
async def proxy_inference(request: InferenceRequest):
    result = await bert_proxy.inference(request.text)
    return {"result": result}


@app.post("/regular_inference")
def regular_inference(request: InferenceRequest):
    result = regular_model.inference(request.text)
    return {"result": result}

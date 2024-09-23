# Batch Inference Toolkit

Batch Inference Toolkit(batch-inference) is a Python package that batches model input tensors coming from multiple requests dynamically, executes the model, un-batches output tensors and then returns them back to each request respectively. This will improve system throughput because of better compute parallelism and better cache locality. The entire process is transparent to developers. 

## When to use

When you want to host Deep Learning model inference on Cloud servers, especially on GPU
It also works for cpu mode too!

## Why to use

It can improve your server throughput up to multiple times

## Advantage of batch-inference

* Platform independent lightweight python library
* Only few lines code change is needed to onboard using built-in
* Flexible APIs to support customized batching algorithms and input types
* Support to avoid python GIL bottleneck
* Tutorials and benchmarks on popular models:

## Benchmark
#### bert-large-uncased

| Concurrency | Regular Way Time | Batch Way Time      |
|-------------|------------------|---------------------|
| 1           | 0.02 sec         | 0.02 sec            |
| 5           | 0.11 sec         | 0.04 sec            |
| 150         | 2.8 - 3.2 sec    | **0.28 - 0.32** sec |

#### bert-base-uncased

| Concurrency | Regular Way Time | Batch Way Time     |
|-------------|------------------|--------------------|
| 1           | meaning less     | meaning less       |
| 100         | 0.78 sec         | 0.07 - 0.1 sec     |
| 150         | 1.10 sec         | **0.1 - 0.15** sec |

## How to Use

```python
class BERTProxy(tessadar.Proxy[str, float]):
    def __init__(self, batch_size=64):
        super().__init__(batch_size=batch_size)
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(self.device)
        self.model.eval()
    
    @override
    async def _inference(self, arguments: list[str]) -> list[float]:
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

    
async def main():
    bert_proxy = BERTProxy()
    # plz be aware that the below example has no underscore '_'
    await bert_proxy.inference("this is a test text.")
```

# GPU Inference Batch Processing with Asyncio

## Overview

The goal is to handle multiple incoming requests by grouping them into batches and processing each batch efficiently using GPU resources. By using `asyncio`, we can handle asynchronous requests and batch them without blocking the main event loop, optimizing GPU utilization and reducing overhead.

## Benefits

- **Efficiency**: Batch processing reduces the overhead per request, making the use of GPU resources more efficient.
- **Scalability**: The system can handle a large number of concurrent requests, which is critical in high-demand environments.
- **Flexibility**: The batch size and other parameters can be adjusted based on the workload and GPU capacity, allowing for dynamic scaling.

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
```

import numpy as np
import onnxruntime as ort
from mosec import Server, Worker
from transformers import AutoTokenizer

MODEL_NAME = "vespa-engine/col-minilm"


class Tokenizer(Worker):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def forward(self, query: str):
        tokens = self.tokenizer(query)
        return (tokens["input_ids"], tokens["attention_mask"])


class ColBERT(Worker):
    def __init__(self):
        self.session = ort.InferenceSession("model_quantized.onnx")

    def forward(self, queries: list[dict]):
        input_ids = np.array(q[0] for q in queries)
        attention_mask = np.array(q[1] for q in queries)
        outputs = self.session.run(
            ["contextual"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )[0]
        return outputs.tolist()


if __name__ == "__main__":
    server = Server()
    server.append_worker(Tokenizer)
    server.append_worker(ColBERT)
    server.run()

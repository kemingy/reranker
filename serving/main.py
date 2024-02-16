from os import environ

from mosec import Server, Worker
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WORKER_NUM = environ.get("WORKER_NUM", 1)


class Encoder(Worker):
    def __init__(self):
        self.model_name = environ.get("MODEL_NAME", DEFAULT_MODEL)
        self.model = CrossEncoder(self.model_name)

    def forward(self, data):
        return self.model.encode(data)


if __name__ == "__main__":
    server = Server()
    server.append_worker(Encoder, num=WORKER_NUM)
    server.run()

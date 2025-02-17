import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, device='cpu'):
        model_path = "/root/hf_models/NV-Embed-v2"
        print(f"Loading embedder model from {model_path}")
        if device == 'cpu':
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        else:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype="float16").to(device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, text, max_length=4096):
        query_embeddings = self.model.encode([text], max_length=max_length)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        return query_embeddings

import json
import tqdm
import torch
from utils.embedder import Embedder


scenarios = [
    "0_test",
    # "1_environment",
    # "2_mining"
    # "3_transport",
    # "4_aerospace", 
    # "5_telecom",
    # "6_architecture",
    # "7_water",
    # "8_farming",
]
embedder = Embedder(device='cpu')

for scenario in scenarios:
    corpus = json.load(open(f"./benchmark/{scenario}/corpus.json"))
    embeddings = []
    for data in tqdm.tqdm(corpus, desc=f"Embedding {scenario} corpus"):
        content = data["content"]
        embeddings.append(embedder.get_embedding(content, max_length=4096))
    embeddings = torch.cat(embeddings)
    print("embeddings.shape", embeddings.shape)

    torch.save(embeddings, f"./benchmark/{scenario}/corpus.pt")
    print(f"Embeddings for {scenario} saved")

print("All done")
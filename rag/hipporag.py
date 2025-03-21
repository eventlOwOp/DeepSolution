from .base import RAGBase
import hipporag


class HippoRAGClient(RAGBase):
    def __init__(self, **kwargs):
        self.hipporag = hipporag.HippoRAG(**kwargs)
        self.hipporag.index(docs=[])

    def prepare(self, docs):
        raise Exception("Not implemented: prepare")

    def retrieve_top_k(self, query, top_k):
        return self.hipporag.retrieve(queries=[query], num_to_retrieve=top_k)[0].docs

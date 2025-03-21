class RAGBase:
    def __init__(self):
        pass

    def prepare(self, docs: list[str]):
        pass

    def retrieve_top_k(self, query, top_k=None) -> list[str]:
        pass

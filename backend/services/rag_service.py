import threading

from rag_system import RAGSystem


class RAGService:
    def __init__(self):
        self._lock = threading.Lock()
        self._instances: dict[str, RAGSystem] = {}

    def get_instance(self, model_preset: str) -> RAGSystem:
        with self._lock:
            if model_preset not in self._instances:
                self._instances[model_preset] = RAGSystem(model_preset=model_preset)
            return self._instances[model_preset]

    def answer(self, query: str, model_preset: str) -> tuple[str, list[str]]:
        rag = self.get_instance(model_preset)
        return rag.answer_query(query)

    def stream(self, query: str, model_preset: str):
        rag = self.get_instance(model_preset)
        yield from rag.stream_query(query)

    def sync_index(self, model_preset: str) -> None:
        rag = self.get_instance(model_preset)
        rag.sync_index()


rag_service = RAGService()

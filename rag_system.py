import os
import re
import yaml
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# LiteLLM-style Model Configuration
# ---------------------------------------------------------------------------
class ModelConfig:
    """Swappable model backend — change model without touching RAG logic."""

    PRESETS = {
        "llama3": {"model": "llama3", "embedding_model": "nomic-embed-text"},
        "mistral": {"model": "mistral", "embedding_model": "nomic-embed-text"},
        "phi3": {"model": "phi3", "embedding_model": "nomic-embed-text"},
        "llama3.1": {"model": "llama3.1", "embedding_model": "nomic-embed-text"},
        "gemma2": {"model": "gemma2", "embedding_model": "nomic-embed-text"},
    }

    def __init__(self, preset: str = "llama3", base_url: str = "http://localhost:11434"):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(self.PRESETS.keys())}")
        cfg = self.PRESETS[preset]
        self.model_name = cfg["model"]
        self.embedding_model = cfg["embedding_model"]
        self.base_url = base_url

    def get_llm(self):
        return OllamaLLM(model=self.model_name, base_url=self.base_url)

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.embedding_model, base_url=self.base_url)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
ENISO_RAW_DATA_DIR = os.path.join(
    BASE_DIR, "EnisoData1-20260418T142956Z-3-001", "EnisoData1"
)


# ---------------------------------------------------------------------------
# RAG System
# ---------------------------------------------------------------------------
class RAGSystem:
    """Enhanced RAG system with metadata filtering, smart chunking, and chat memory."""

    # Category keywords for auto-routing queries
    CATEGORY_KEYWORDS = {
        "timetable": [
            "emploi", "temps", "schedule", "timetable", "groupe", "group",
            "cours", "class", "salle", "room", "horaire",
        ],
        "calendar": [
            "calendrier", "calendar", "examen", "exam", "ds", "devoir",
            "surveillé", "date",
        ],
        "stage": [
            "stage", "internship", "entreprise", "company", "rapport",
            "report", "été", "summer", "lettre", "appui",
        ],
        "pfe": [
            "pfe", "projet", "fin", "études", "graduation", "project",
            "sujet", "topic", "encadrant", "supervisor",
        ],
        "inscription": [
            "inscription", "réinscription", "enroll", "registration",
            "inscrire",
        ],
    }

    def __init__(
        self,
        data_dir_path: str | None = None,
        db_path: str = "chroma",
        model_preset: str = "llama3",
    ) -> None:
        print("🚀 Initialisation du système RAG ENISO (Enhanced)...")

        self.data_directory = data_dir_path or PROCESSED_DATA_DIR
        self.db_path = db_path
        self.model_config = ModelConfig(preset=model_preset)
        self.model = self.model_config.get_llm()

        # Chat memory: list of (role, message) tuples
        self.chat_history: list[tuple[str, str]] = []
        self.max_history = 6  # keep last 6 exchanges

        # System prompt
        self.system_prompt = (
            "Tu es un assistant IA spécialisé pour l'ENISO "
            "(École Nationale d'Ingénieurs de Sousse). "
            "Tu réponds de manière précise, concise et utile en utilisant "
            "UNIQUEMENT le contexte fourni. "
            "Si l'information n'est pas dans le contexte, dis-le honnêtement. "
            "Réponds dans la même langue que la question (Arabe, Français ou Anglais). "
            "Quand tu mentionnes des horaires ou des groupes, sois très précis."
        )

        self.prompt_template = """
{system_prompt}

{chat_history_section}

Contexte extrait des documents officiels de l'ENISO :
---
{context}
---

Question : {question}

Réponse :
"""

        # Build the vector store
        if not os.path.exists(self.data_directory):
            print(f"⚠ Data directory not found: {self.data_directory}")
            print("  Run `python preprocess_data.py` first to generate processed data.")
            print("  Falling back to raw PDF data...")
            self.data_directory = ENISO_RAW_DATA_DIR
            self._setup_collection_pdf()
        else:
            self._setup_collection()

    # ------------------------------------------------------------------
    # Document loading (processed markdown files)
    # ------------------------------------------------------------------
    def _load_processed_documents(self) -> list[Document]:
        """Load preprocessed markdown files with embedded metadata."""
        documents = []
        md_files = sorted(Path(self.data_directory).glob("*.md"))

        for md_path in md_files:
            raw = md_path.read_text(encoding="utf-8")

            # Parse YAML frontmatter
            meta = {}
            content = raw
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    try:
                        meta = yaml.safe_load(parts[1]) or {}
                    except yaml.YAMLError:
                        pass
                    content = parts[2].strip()

            meta["source"] = str(md_path)
            documents.append(Document(page_content=content, metadata=meta))

        return documents

    # ------------------------------------------------------------------
    # Smart chunking with markdown header awareness
    # ------------------------------------------------------------------
    def _smart_split(self, documents: list[Document]) -> list[Document]:
        """Two-stage splitting: first by markdown headers, then by size."""

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
                ("###", "subsection"),
            ],
            strip_headers=False,
        )

        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        all_chunks = []
        for doc in documents:
            # Stage 1: split by headers
            try:
                header_chunks = header_splitter.split_text(doc.page_content)
            except Exception:
                header_chunks = [doc]

            for hchunk in header_chunks:
                # Merge parent metadata
                if isinstance(hchunk, Document):
                    merged_meta = {**doc.metadata, **hchunk.metadata}
                    hchunk_text = hchunk.page_content
                else:
                    merged_meta = dict(doc.metadata)
                    hchunk_text = str(hchunk)

                # Stage 2: split large chunks by character count
                if len(hchunk_text) > 1200:
                    sub_docs = size_splitter.create_documents(
                        [hchunk_text], metadatas=[merged_meta]
                    )
                    all_chunks.extend(sub_docs)
                else:
                    all_chunks.append(
                        Document(page_content=hchunk_text, metadata=merged_meta)
                    )

        return all_chunks

    # ------------------------------------------------------------------
    # Chunk ID generation
    # ------------------------------------------------------------------
    def _assign_chunk_ids(self, chunks: list[Document]) -> list[Document]:
        """Assign unique IDs based on source + index."""
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source_file", "unknown")
            chunk.metadata["chunk_id"] = f"{source}_{i}"
        return chunks

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------
    def _setup_collection(self):
        """Load processed data, chunk, and upsert into ChromaDB."""
        documents = self._load_processed_documents()
        if not documents:
            print("⚠ No processed documents found. Run preprocess_data.py first.")
            return

        print(f"📄 Loaded {len(documents)} processed documents")
        chunks = self._smart_split(documents)
        chunks = self._assign_chunk_ids(chunks)
        print(f"🔪 Split into {len(chunks)} chunks")

        vectordb = self._initialize_vectorDB()
        existing = vectordb.get()
        existing_ids = set(existing["ids"])
        print(f"📦 Existing chunks in DB: {len(existing_ids)}")

        new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]

        if new_chunks:
            # Batch embedding to avoid timeouts — process in groups of 50
            batch_size = 50
            total = len(new_chunks)
            for i in range(0, total, batch_size):
                batch = new_chunks[i : i + batch_size]
                batch_ids = [c.metadata["chunk_id"] for c in batch]
                vectordb.add_documents(batch, ids=batch_ids)
                done = min(i + batch_size, total)
                print(f"  📥 Embedded {done}/{total} chunks...")
            print(f"✅ Added {total} new chunks to the database")
        else:
            print("✅ Database is up to date — no new chunks to add.")

    def _setup_collection_pdf(self):
        """Fallback: load raw PDFs (old behavior)."""
        from langchain_community.document_loaders import PyPDFDirectoryLoader

        loader = PyPDFDirectoryLoader(self.data_directory, glob="**/*.pdf")
        pages = loader.load()
        if not pages:
            print("⚠ No PDF documents found.")
            return

        print(f"📄 Loaded {len(pages)} PDF pages (fallback mode)")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150, length_function=len
        )
        chunks = splitter.split_documents(pages)

        # Assign IDs
        for i, c in enumerate(chunks):
            src = c.metadata.get("source", "unknown")
            page = c.metadata.get("page", 0)
            c.metadata["chunk_id"] = f"{src}_{page}_{i}"
            c.metadata["category"] = "general"

        vectordb = self._initialize_vectorDB()
        existing_ids = set(vectordb.get()["ids"])
        new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]

        if new_chunks:
            batch_size = 50
            total = len(new_chunks)
            for i in range(0, total, batch_size):
                batch = new_chunks[i : i + batch_size]
                batch_ids = [c.metadata["chunk_id"] for c in batch]
                vectordb.add_documents(batch, ids=batch_ids)
                done = min(i + batch_size, total)
                print(f"  📥 Embedded {done}/{total} chunks...")
            print(f"✅ Added {total} chunks (fallback)")

    # ------------------------------------------------------------------
    # Query routing: detect category from question
    # ------------------------------------------------------------------
    def _detect_query_category(self, query: str) -> str | None:
        """Detect the most likely category from the query text."""
        query_lower = query.lower()
        scores = {}
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[cat] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    # ------------------------------------------------------------------
    # Context retrieval with optional category filter
    # ------------------------------------------------------------------
    def _retrieve_context(self, query: str, k: int = 5, score_threshold: float = 1.5):
        """Retrieve relevant chunks, optionally filtered by category."""
        vectordb = self._initialize_vectorDB()
        category = self._detect_query_category(query)

        if category:
            print(f"🏷️  Detected category: {category}")
            # Try filtered search first
            try:
                results = vectordb.similarity_search_with_score(
                    query,
                    k=k,
                    filter={"category": category},
                )
                if results:
                    # Also add some unfiltered results for broader context
                    unfiltered = vectordb.similarity_search_with_score(query, k=2)
                    seen_ids = {doc.metadata.get("chunk_id") for doc, _ in results}
                    for doc, score in unfiltered:
                        if doc.metadata.get("chunk_id") not in seen_ids:
                            results.append((doc, score))
                    return results
            except Exception:
                pass  # Fall through to unfiltered

        # Unfiltered retrieval
        results = vectordb.similarity_search_with_score(query, k=k)

        # Filter by score threshold (lower is better in Chroma's L2 distance)
        if results:
            filtered = [(doc, score) for doc, score in results if score < score_threshold]
            if filtered:
                return filtered

        return results

    # ------------------------------------------------------------------
    # Chat history management
    # ------------------------------------------------------------------
    def _format_chat_history(self) -> str:
        if not self.chat_history:
            return ""

        lines = ["Historique de conversation récent :"]
        for role, msg in self.chat_history[-self.max_history :]:
            prefix = "Utilisateur" if role == "user" else "Assistant"
            # Truncate long messages in history
            short_msg = msg[:200] + "..." if len(msg) > 200 else msg
            lines.append(f"{prefix}: {short_msg}")
        return "\n".join(lines)

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history.clear()

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------
    def answer_query(self, query_text: str) -> tuple[str, list[str]]:
        """Generate an answer with context retrieval and chat memory."""
        prompt, context_results = self._build_prompt(query_text)
        response_text = self.model.invoke(prompt)
        sources = self._finalize_response(query_text, response_text, context_results)
        return response_text, sources

    def stream_query(self, query_text: str):
        """Stream an answer token-by-token. Yields (token, None) during generation,
        then (None, sources) at the end."""
        prompt, context_results = self._build_prompt(query_text)

        full_response = []
        for chunk in self.model.stream(prompt):
            full_response.append(chunk)
            yield chunk, None

        response_text = "".join(full_response)
        sources = self._finalize_response(query_text, response_text, context_results)
        yield None, sources

    def _build_prompt(self, query_text: str):
        """Retrieve context and build the prompt."""
        context_results = self._retrieve_context(query_text)

        print(f"\n***** CONTEXTE RÉCUPÉRÉ ({len(context_results)} chunks) ******")
        for doc, score in context_results:
            src = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
            cat = doc.metadata.get("category", "?")
            print(f"  [score={score:.3f}] [{cat}] {src}")

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _ in context_results]
        )

        chat_history_section = self._format_chat_history()

        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(
            system_prompt=self.system_prompt,
            chat_history_section=chat_history_section,
            context=context_text,
            question=query_text,
        )
        return prompt, context_results

    def _finalize_response(self, query_text, response_text, context_results):
        """Update chat history and return sources."""
        self.chat_history.append(("user", query_text))
        self.chat_history.append(("assistant", response_text))

        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2 :]

        sources = list({
            f"{doc.metadata.get('source_file', os.path.basename(doc.metadata.get('source', 'unknown')))} "
            f"[{doc.metadata.get('category', '?')}]"
            for doc, _ in context_results
        })
        return sources

    # ------------------------------------------------------------------
    # Vector DB
    # ------------------------------------------------------------------
    def _initialize_vectorDB(self):
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.model_config.get_embeddings(),
        )

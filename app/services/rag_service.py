from pathlib import Path

from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_loader import load_pdf
from app.llm.generator import Generator
from app.vectorstore.faiss_store import FAISSVectorStore


class RAGService:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = FAISSVectorStore(dimension=768)
        self.generator = Generator()

        # Directory where the FAISS index and documents are stored
        self.vector_dir = Path("vectorstore")
        self.vector_dir.mkdir(exist_ok=True)

        # Load an existing index if available
        loaded = self.vector_store.load(self.vector_dir)

        if loaded:
            print(
                f"Loaded existing vector index with {len(self.vector_store.documents)} chunks."
            )
        else:
            print("No existing vector index found. Starting fresh.")

    def index_pdf(self, pdf_path: Path) -> dict:
        documents = load_pdf(pdf_path)
        chunks = chunk_documents(documents)

        embeddings = self.embedder.embed_documents(
            [doc.content for doc in chunks]
        )

        self.vector_store.add_documents(chunks, embeddings)

        # Persist the updated index
        self.vector_store.save(self.vector_dir)

        return {
            "pages_processed": len(documents),
            "chunks_created": len(chunks),
        }

    def ask(self, question: str) -> dict:
        if len(self.vector_store.documents) == 0:
            return {
                "answer": "No documents have been indexed yet.",
                "sources": [],
            }

        query_embedding = self.embedder.embed_text(question)

        results = self.vector_store.search(query_embedding, k=3)

        if not results:
            return {
                "answer": "I could not find relevant information in the uploaded documents.",
                "sources": [],
            }

        context = "\n\n".join(
            [doc.content for doc, _ in results]
        )

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        sources = []

        for doc, _ in results:
            sources.append(
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }
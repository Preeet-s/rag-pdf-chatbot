from app.ingestion.document import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    chunks: list[Document] = []

    for document in documents:
        text = document.content

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            metadata = document.metadata.copy()
            metadata["chunk_id"] = chunk_id

            chunks.append(
                Document(
                    content=chunk_text,
                    metadata=metadata,
                )
            )

            start += chunk_size - chunk_overlap
            chunk_id += 1

    return chunks
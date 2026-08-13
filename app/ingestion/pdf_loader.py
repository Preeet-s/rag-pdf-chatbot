from pathlib import Path

from pypdf import PdfReader

from app.ingestion.document import Document


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Load a PDF file and return one Document object per page.
    """

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        metadata = {
            "source": pdf_path.name,
            "page": page_number,
        }

        document = Document(
            content=text,
            metadata=metadata,
        )

        documents.append(document)

    return documents


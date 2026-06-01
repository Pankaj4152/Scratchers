from pathlib import Path 
from pydoc import Doc
from typing import List

from pypdf import PdfReader
from docx import Document as DocxReader


from ingestion.document import Document
from ingestion.cleaner import TextCleaner

class FileLoader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.cleaner = TextCleaner(remove_extra_whitespace=True, lowercase=False)
        self._parsers = {
            ".txt": self._load_txt,
            ".md": self._load_txt,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
        }

    def load(self) -> List[Document]:
        """
        Load a file and return a list of Document objects.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"File not found: {self.path}"
            )
        
        ext = self.path.suffix.lower()
        if ext not in self._parsers:
            raise ValueError(
                f"Unsupported file type: {ext}"
            )
        
        raw_text = self._parsers[ext]()
        cleaned_text = self.cleaner.clean(raw_text)
        return [
            Document(
                text=cleaned_text,
                metadata={
                    "source": str(self.path),
                    "type": ext.lstrip("."),
                    "file_name": self.path.name,
                }
            )
        ]

    def _load_txt(self) -> str:
        """Reads plain text and markdown files."""
        return self.path.read_text(encoding="utf-8")
    
    def _load_docx(self) -> str:
        """Reads Microsoft Word documents."""
        doc = DocxReader(self.path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _load_pdf(self) -> str:
        """Reads PDF files page by page."""
        reader = PdfReader(self.path)
        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)
        return "\n".join(extracted_pages)
    



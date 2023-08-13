from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text_chunks = []
        for page in reader.pages[:1]:
            text_chunks.append(page.extract_text())

    return "\n".join(text_chunks)

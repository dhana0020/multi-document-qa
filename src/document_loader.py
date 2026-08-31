import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF while preserving
    document name and page number.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "document": Path(pdf_path).name,
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


def load_all_pdfs(folder_path):
    """
    Load all PDF files from a folder.
    """

    folder = Path(folder_path)

    all_pages = []

    pdf_files = list(folder.glob("*.pdf"))

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pages = extract_text_from_pdf(pdf_file)

        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":

    folder = Path("data/documents")

    pages = load_all_pdfs(folder)

    print("\n--------------------------------")
    print("PDF INGESTION COMPLETED")
    print("--------------------------------")

    print(f"Total pages extracted: {len(pages)}")

    for page in pages[:3]:
        print("\nDocument:", page["document"])
        print("Page:", page["page"])
        print("Text:", page["text"][:300])
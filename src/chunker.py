
import json

def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Convert page-level documents into smaller chunks
    while preserving document and page metadata.
    """

    all_chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0
        chunk_id = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            all_chunks.append({
                "document": page["document"],
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            chunk_id += 1

            start += chunk_size - overlap

    return all_chunks






if __name__ == "__main__":

    from document_loader import load_all_pdfs
    from pathlib import Path

    folder = Path("data/documents")

    pages = load_all_pdfs(folder)

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100
    )

    output_path = Path("data/processed/chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n--------------------------------")
    print("CHUNKING COMPLETED")
    print("--------------------------------")

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved to: {output_path}")
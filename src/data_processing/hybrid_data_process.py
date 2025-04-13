import os
import fitz  # PyMuPDF
import json
import re
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# TEXT EXTRACTION UTILITIES

def extract_text_from_pdf(pdf_path):
    """Extracts and segments text from a PDF with sections/subsections and page numbers."""
    text_data = []
    current_section = ""
    current_subsection = ""

    section_pattern = re.compile(r'^(\d+(\.\d+)*)\s+(.+)$')

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            cleaned_text = clean_text(text)
            lines = cleaned_text.split(". ")

            for line in lines:
                match = section_pattern.match(line)
                if match:
                    if len(match.group(1).split('.')) == 1:
                        current_section = match.group(3)
                        current_subsection = ""
                    else:
                        current_subsection = match.group(3)

                if line.strip():
                    text_data.append({
                        "page": page_num,
                        "section": current_section if current_section else "Uncategorized",
                        "subsection": current_subsection if current_subsection else "Uncategorized",
                        "text": line.strip()
                    })

    return text_data

def clean_text(text):
    """Removes excessive whitespace and line breaks."""
    return re.sub(r'\s+', ' ', text).strip()

# MAIN HYBRID INDEXING LOGIC

def process_pdfs_and_build_hybrid_index(pdf_folder, index_output_path, metadata_output_path, model_name="all-MiniLM-L6-v2"):
    all_passages = []
    model = SentenceTransformer(model_name)

    print("Scanning PDF files in:", pdf_folder)
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"Extracting from: {filename}")
            extracted = extract_text_from_pdf(file_path)

            for passage in extracted:
                passage["document"] = filename
                if passage["text"]:
                    all_passages.append(passage)

    print(f"📚 Total extracted passages: {len(all_passages)}")

    # Generate embeddings and normalize for cosine similarity
    texts = [p["text"] for p in all_passages]
    print("Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    print(f"FAISS index saved to: {index_output_path}")

    # Save enriched metadata
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_passages, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved to: {metadata_output_path}")

# EXECUTION ENTRY POINT

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    pdf_input_dir = os.path.join(project_root, 'data', 'raw')
    index_output = os.path.join(project_root, 'models', 'index', 'index_files', 'faiss_index')
    metadata_output = os.path.join(project_root, 'models', 'index', 'index_files', 'metadata_passages.json')

    # Vérifie si le dossier des PDF existe
    if not os.path.exists(pdf_input_dir):
        raise FileNotFoundError(f"Input PDF folder not found: {pdf_input_dir}")

    process_pdfs_and_build_hybrid_index(pdf_input_dir, index_output, metadata_output)


# PDF Chat (Local LLM 🤗)

This is a quick demo of showing how to create an LLM-powered document Q&A application using LangChain and open-source LLMs.
It uses [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for embedding, and [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) for text summarization.

[demo.webm](https://github.com/liminma/pdfChat/assets/47096483/2b1e5942-4bec-47a2-90a0-7fa75163cb1f)

## Implementaion
- PDF ingestion and chunking.
  - use `PyMuPDF` to extract texts from PDF file.
  - chunking is done based on the unit of paragraph.
    - filter out paragraphs that are shorter than certain number of characters (default to 10 chars).
    - metadata of page number and bounding box of the paragraph are kept for highlighting the paragraph.
- use Chroma as the embedding database.
- similarity search results are passed to LLM for summarization.

## How to run
To build a Docker image:
```bash
docker build --tag pdfchat .
```

To start a container:
```bash
# with GPU
docker run --init -p 8501:8501 --gpus all pdfchat

# no GPU
docker run --init -p 8501:8501 pdfchat
```
Then view the application in a browser: http://localhost:8501

# PDF Chat (Llama 2 🤗)

This is a quick demo of showing how to create an LLM-powered PDF Q&A application using LangChain and Meta Llama 2.
It uses [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for embedding, and [Meta Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) for question answering.

&nbsp;

[demo.webm](https://github.com/liminma/pdfChat/assets/47096483/2b1e5942-4bec-47a2-90a0-7fa75163cb1f)

&nbsp;

## Implementaion
- PDF ingestion and chunking.
  - use `PyMuPDF` to extract texts from PDF file.
  - Each chunk consists of one or more PDF blocks. The default minimum chunk length is 1000 chars.
  - metadata: starting page number and the bounding boxes of the contained blocks.
    - use bounding box to highlight a block.
- use Chroma as the embedding database.
- similarity search results (chunks) are passed to LLM as context for answering a user question.

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

import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class PDFChatBot:
    def __init__(self, 
                 pdf_file: str,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 chunking_separators = None
                ):
        self.pdf_file = pdf_file
        self._chunk_size = chunk_size if chunk_size else 1500
        self._chunk_overlap = chunk_overlap if chunk_overlap else 150
        self._chunking_separators = chunking_separators if chunking_separators else ["\n\n", "\n", "(?<=\. )", " ", ""]

        # hard code the embedding model and summarization model checkpoint
        self._emb_model = 'sentence-transformers/all-mpnet-base-v2'
        self._llm_checkpoint = 'facebook/bart-large-cnn'
        
    
    def embed_pdf(self):
        # load the pdf file first
        pdf_loader = PyPDFLoader(self.pdf_file)
        self._pdf_pages = pdf_loader.load()

        # use RecursiveCharacterTextSplitter for splitting generic text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._chunking_separators
        )
        self._docs = text_splitter.split_documents(self._pdf_pages)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding = HuggingFaceEmbeddings(
            model_name=self._emb_model,
            model_kwargs={'device': device}
        )
        
        # use Chroma as vector db, not persisting the embeddings
        self.vectordb = Chroma.from_documents(
            documents=self._docs,
            embedding=embedding,
            persist_directory=None
        )
    
    
    def load_llm(self):
        pass
    
    
    def search(self):
        pass
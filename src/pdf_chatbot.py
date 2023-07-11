import torch
from transformers import pipeline

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 150

# hard code the names of embedding model and summarization model.
# they are good enough for this application.
DEFAULT_EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEFAULT_MODEL = 'facebook/bart-large-cnn'


def load_llm():
    """
    load the summarization model using the HuggingFace pipeline,
    use GPU if possible
    """
    return pipeline(task="summarization", model=DEFAULT_MODEL, device_map='auto')


def load_emb():
    """
    load the embedding model,
    use GPU if possible
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBED_MODEL,
        model_kwargs={'device': device}
    )
    
    
class PDFChatBot:
    def __init__(self) -> None:
        self.llm = None
        self.embedding = None
        self.vectordb = None
        

    def load_vectordb(self,
                      pdf_filepath: str,
                      chunk_size: int = None,
                      chunk_overlap: int = None,
                      chunking_separators: list[str] = None) -> None:
        if pdf_filepath is None:
            raise ValueError("Param 'pdf_filepath' can not be None.")

        # do some validation for chunking configurations
        chunk_size = chunk_size if chunk_size and chunk_size > 0 else 1500
        chunk_overlap = chunk_overlap if chunk_overlap and chunk_overlap > 0 else 150
        chunking_separators = chunking_separators or ["\n\n", "\n", "(?<=\. )", " ", ""]
        
        # load the pdf file first
        self._pdf_pages = PyPDFLoader(pdf_filepath).load()

        # use RecursiveCharacterTextSplitter for splitting generic text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=chunking_separators
        )
        self._docs = text_splitter.split_documents(self._pdf_pages)
        
        # use Chroma as vector db.
        # no need to persist the database
        self.vectordb = Chroma.from_documents(
            documents=self._docs,
            embedding=self.embedding,
            persist_directory=None
        )
    
    
    def mmr_search(self, query, k=5, fetch_k=7, max_length=130, min_length=30):
        """This method uses maximal marginal relevance (mmr) to search the vector database, 
        then summarizes each chunk, and finally summarizes the summaries of all chunks from
        the previous step.
        """
        # retrieve relevant chunks from the vector database
        src_docs = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

        # summarize each chunk
        texts = [doc.page_content for doc in src_docs]
        sub_summaries = self.llm(texts, max_length=max_length, min_length=min_length, do_sample=False)
        
        # do a final summarization based on sub_summaries
        texts = [ss['summary_text'] for ss in sub_summaries]
        texts = '\n\n'.join(texts)
        summary = self.llm(texts, max_length=max_length*2, min_length=min_length*2, do_sample=False)

        return summary, src_docs, sub_summaries
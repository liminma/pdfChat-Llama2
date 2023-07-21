import fitz

import torch
from transformers import pipeline
from transformers import GenerationConfig

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document

from .prompts import llama2_template, llama2_prompt_ending_words


# hard code the names of models.
# they are good enough for this demo application.
DEFAULT_EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DEFAULT_MODEL = 'meta-llama/Llama-2-7b-chat-hf'


def load_llm() -> (pipeline, str):
    """
    load the model using the HuggingFace pipeline, use GPU if possible.
    """
    gen_config = GenerationConfig.from_pretrained(DEFAULT_MODEL)
    gen_config.max_new_tokens = 1000
    
    llm = pipeline(
        task="text-generation",
        model=DEFAULT_MODEL,
        torch_dtype=torch.float16,
        generation_config=gen_config,
        device_map='auto',
    )
    
    # load the corresponding prompt template
    llama2_prompt = PromptTemplate.from_template(llama2_template)
    
    return llm, llama2_prompt, llama2_prompt_ending_words


def load_emb() -> HuggingFaceEmbeddings:
    """
    load the embedding model,
    use GPU if possible
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBED_MODEL,
        model_kwargs={'device': device}
    )


def split_pdf_blocks(pdf_bytes: bytes, filename: str = None, min_length: int = 10) -> list[Document]:
    """Split a PDF file into a list of blocks. Each block's meta data
    contain its page number and bounding box of the block.

    Parameters:
        pdf_bytes: the pdf file in bytes.
        filename: the name of the uploaded file.
        min_length: the min length of a block to keep, default to 10 chars.

    Return:
        a list of `langchain.schema.document.Document`.
    """
    pdf_doc = fitz.open("pdf", pdf_bytes)

    documents = []
    for i, page in enumerate(pdf_doc):
        blocks = page.get_text('blocks')
        for block in blocks:
            page_content = block[4]
            if len(page_content) < min_length:
                continue

            metadata = {
                'source': filename if filename else 'dummy name', # not used in the current impl.
                'page': i,
                'bbox': ','.join(map(str, block[:4])), # langChain only allows str, int or float for meta data values
            }
            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

    return documents


class PDFChatBot:
    def __init__(self) -> None:
        self.embedding = None
        self.llm = None
        self.prompt =None
        self.prompt_ending_words = None
        self.vectordb = None


    def load_vectordb(self, docs: list[Document]) -> None:
        """create a vector db from input documents

        Parameters:
            docs: a list of LangChain documents.
        """
        self._docs = docs

        # use Chroma as vector db.
        # no need to persist the database
        self.vectordb = Chroma.from_documents(
            documents=self._docs,
            embedding=self.embedding,
            persist_directory=None
        )


    def mmr_search(self,
                   query: str,
                   k: int = 5,
                   fetch_k: int = 7,
                   max_length: int = 130,
                   min_length: int = 30) -> (list[dict], list[Document], list[dict]):
        """This method uses maximal marginal relevance (mmr) to search the vector database,
        then summarizes each chunk, and finally summarizes the summaries of all chunks from
        the previous step.

        Parameters:
            query: input question.
            k: the number of returned documents (default to 5).
            fetch_k: the number of documents to fetch to pass to MMR algorithm (default to 7).
            max_length: max length to pass to model.
            min_length: min length to pass to model.

        Return:
            a tuple of summary, found relevant documents, summaries for each document.
        """
        # retrieve relevant chunks from the vector database
        src_docs = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

        # construct context
        texts = [doc.page_content for doc in src_docs]
        ctx = '\n\n'.join(texts)
        
        seqs = self.llm(
            self.prompt.format(context=ctx, question=query),
            do_sample=True,        
            num_return_sequences=1,
        )
        answer = seqs[0]['generated_text']
        idx_start = answer.index(self.prompt_ending_words) + len(self.prompt_ending_words)
        answer = answer[idx_start:].strip()
        
        return answer, src_docs

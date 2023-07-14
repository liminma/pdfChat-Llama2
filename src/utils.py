import fitz
import base64

from langchain.schema.document import Document


def highlight_block(src_doc: Document, pdf_bytes: bytes) -> str:
    """Highlight a block (paragraph) in a PDF page
    
    Parameters:
        src_doc: a LangChain document containing metadata of the block.
        pdf_bytes: the PDF file in bytes.
        
    Return:
        pdf_base64: the modified PDF file with highlighted block in base64.
        page_num: the page containing the block.
    """
    # the page number containing the block
    page_num = src_doc.metadata['page']
    
    # get the bounding box of the block to be highlighted
    bbox = list(map(float, src_doc.metadata['bbox'].split(',')))
    bbox = fitz.fitz.Rect(bbox)

    pdf_doc = fitz.open("pdf", pdf_bytes)
    # get the page containing the block
    page = pdf_doc[page_num]
    # highlight the block using the bounding box
    hl = page.add_highlight_annot(bbox)
    hl.update()

    # convert using base64 encoding
    pdf_doc = pdf_doc.tobytes(
        deflate=True,
        clean=True,
        encryption=fitz.PDF_ENCRYPT_KEEP
    )
    pdf_base64 = base64.b64encode(pdf_doc).decode('utf-8')
    
    return pdf_base64, page_num


def get_pdf_view(pdf_base64, page_num: int = 0) -> str:
    """create an iframe to embed PDF file
    
    Parameters:
        pdf_base64: PDF file encoded using base64.
        page_num: the page to dispaly.
        
    Return:
        HTML snippet of iframe
    """
    return (f'''
    <iframe src="data:application/pdf;base64,{pdf_base64}#page={page_num}"
    width="100%" height="1000px" type="application/pdf" style="min-width:400px;"></iframe>
    ''')


def set_background_image(img_base64):
    """set a base64 encoded image as page background.
    Parameters:
        img_base64: an image encoded in base64.    
    Return:
        HTML snippet of background image.
    """
    return '''
    <style>
      .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: repeat;
      }
    </style>
    ''' % img_base64
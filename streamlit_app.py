import base64
import streamlit as st

import src.pdf_chatbot as pdf_chatbot
from src.pdf_chatbot import PDFChatBot
from src.utils import highlight_block, get_pdf_view, set_background_image


@st.cache_resource
def load_llm():
    return pdf_chatbot.load_llm()


@st.cache_resource
def load_embing():
    return pdf_chatbot.load_emb()


@st.cache_resource()
def base64_encoding(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        data = f.read()        
    return base64.b64encode(data).decode()


st.set_page_config(
    page_title='PDF Chat (Local LLM)',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='auto',
)

with open( "css/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

html_background =  set_background_image(
    base64_encoding('assets/background.png')
)
st.markdown(html_background, unsafe_allow_html=True)

footer="""
<div class="footer">
  <p>background photo by
     <a href="https://unsplash.com/@olga_o?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Olga Thelavart</a> 
     on
     <a href="https://unsplash.com/photos/HZm2XR0whdw?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  </p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
    
# use HTML snippets for title and sub-title in order to apply custom CSS rules
html_title = '<p id="title">PDF Chat</p>'
html_subtitle = '<div id="subtitle-container"><span id="subtitle">powered by local LLMs </span><span id="hfemoji">🤗</span></div>'
st.markdown(html_title, unsafe_allow_html=True)
st.markdown(html_subtitle, unsafe_allow_html=True)

st.divider()

if 'chatbot' not in st.session_state:
    st.session_state.chatbot : PDFChatBot = PDFChatBot()

col1, col2 = st.columns([0.6, 0.4])

if 'file_updated' not in st.session_state:
    st.session_state.file_updated = True
    
def file_updated():
    st.session_state.file_updated = True
    st.session_state.pdf_bytes = None
    st.session_state.src_docs = None
    st.session_state.answer = None
    
with col1:
    pdf_file = st.file_uploader('**Upload a file (PDF only)**', type='pdf', on_change=file_updated)
    if pdf_file is None:
        st.stop()

    pdf_placeholder = st.empty()
        
    if st.session_state.file_updated:
        with st.spinner(text=f'processing {pdf_file.name} ...'):
            st.session_state.pdf_bytes = pdf_file.read()
            docs = pdf_chatbot.split_pdf_blocks(st.session_state.pdf_bytes, filename=pdf_file.name)

            st.session_state.chatbot.embedding = load_embing()
            st.session_state.chatbot.llm = load_llm()
            st.session_state.chatbot.load_vectordb(docs)
        st.session_state.file_updated = False

    base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
    pdf_view = get_pdf_view(base64_pdf)
    pdf_placeholder.markdown(pdf_view, unsafe_allow_html=True)

with col2:
    question = st.text_input('Question:', '')
    if st.button('Answer'):
        try:
            answer, src_docs, _ = st.session_state.chatbot.mmr_search(question)
            st.session_state.src_docs = src_docs
            st.session_state.answer = f'{answer[0]["summary_text"]}'
        except Exception as ex:
            st.error(f'Errors: {str(ex)}')
    
    if st.session_state.answer:
        st.write(st.session_state.answer)
    
    if st.session_state.src_docs is not None:
        nested_cols = st.columns(len(st.session_state.src_docs))

        for i, src_doc in enumerate(st.session_state.src_docs):
            with nested_cols[i]:
                if st.button(f'source #{i+1}'):
                    pdf_base64, page_num = highlight_block(src_doc, st.session_state.pdf_bytes)
                    pdf_view = get_pdf_view(pdf_base64, page_num=page_num+1)
                    pdf_placeholder.markdown(pdf_view, unsafe_allow_html=True)

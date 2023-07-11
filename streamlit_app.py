import shutil
import streamlit as st
from tempfile import NamedTemporaryFile

import src.pdf_chatbot as pdf_chatbot
from src.pdf_chatbot import PDFChatBot


title = 'PDF Chat'
st.set_page_config(
    page_title=title,
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='auto',
)

if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] : PDFChatBot = PDFChatBot()

@st.cache_resource
def load_llm():
    return pdf_chatbot.load_llm()

@st.cache_resource
def load_emb():
    return pdf_chatbot.load_emb()

st.title(title)

with st.sidebar:
    pdf_file = st.file_uploader('**Upload your paper (PDF only)**', type='pdf')

    if st.button('Upload') and pdf_file is not None:
        with st.spinner(text='Uploading ...'):
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                shutil.copyfileobj(pdf_file, tmp)

            st.session_state['chatbot'].embedding = load_emb()
            st.session_state['chatbot'].llm = load_llm()
            st.session_state['chatbot'].load_vectordb(tmp.name)
            st.sidebar.success('PDF uploaded successfully')

question = st.text_input('Question:', '')

if st.button('Answer'):
    try:
        answer, _, _ = st.session_state['chatbot'].mmr_search(question)
        st.write(f'{answer[0]["summary_text"]}')
    except Exception as ex:
        st.error(f'Errors: {str(ex)}')
import streamlit as st
from constant import TEMPLATES, MAX_RETRIEVAL_DOCS
from config import DB_NAME, COLLECTION_NAME, INDEX_NAME
from db import init_mongo_connection, clear_document_data
from handlers import SmartStreamHandler
from pdf_utils import extract_text_from_pdf, get_file_hash, save_uploaded_file
from llm_utils import get_vectorstore, is_relevant
from ui import show_thinking_animation, render_chat_history
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

def main():
    st.set_page_config(page_title="⚡ Smart PDF Chat (MongoDB)", layout="wide", page_icon="⚡")
    client, err = init_mongo_connection()
    if err:
        st.error(f"{TEMPLATES['db_error']}: {err}")
        st.stop()
    else:
        st.toast(TEMPLATES["db_connected"], icon="✅")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.initial_load = True

    if "processed_pdf_hash" not in st.session_state:
        st.session_state.processed_pdf_hash = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.title("⚡ Intelligent Document Assistant with MongoDB Atlas")

    with st.sidebar:
        model_name = st.selectbox("Ollama Model", ["deepseek-r1:1.5b", "llama3.2", "llama2"], index=0)
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.3, 0.1)

        if st.session_state.vectorstore and st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.initial_load = True
            st.rerun()

        if st.session_state.processed_pdf_hash and st.button("Remove Document from DB"):
            clear_document_data(client, DB_NAME, COLLECTION_NAME, st.session_state.processed_pdf_hash)
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.success("Document removed from database!")
            st.rerun()

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file:
        file_hash = get_file_hash(uploaded_file)
        uploaded_file.seek(0)
        if st.session_state.processed_pdf_hash != file_hash:
            with st.spinner(TEMPLATES["processing"]):
                st.session_state.processed_pdf_hash = file_hash
                st.session_state.chat_history = []
                st.session_state.initial_load = True
                raw_text = extract_text_from_pdf(uploaded_file)
                if raw_text:
                    collection = client[DB_NAME][COLLECTION_NAME]
                    st.session_state.vectorstore = get_vectorstore(raw_text, file_hash, collection, INDEX_NAME)
                    st.success(TEMPLATES["ready"])

    if st.session_state.vectorstore:
        if st.session_state.initial_load:
            show_thinking_animation()
            st.session_state.initial_load = False
        render_chat_history(st.session_state.chat_history)

        if prompt := st.chat_input("Ask about the document"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if not is_relevant(prompt, st.session_state.vectorstore):
                    st.markdown(TEMPLATES["irrelevant"])
                    st.session_state.chat_history.append({"role": "assistant", "content": TEMPLATES["irrelevant"]})
                else:
                    container = st.empty()
                    handler = SmartStreamHandler(container)

                    llm = Ollama(model=model_name, temperature=temperature, callbacks=[handler])
                    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": MAX_RETRIEVAL_DOCS, "score_threshold": 0.4})

                    prompt_template = PromptTemplate(
                        template="""Answer the question based on the document content below.
                        1. Summary Paragraph
                        2. 3–5 bullet points
                        3. Important details/numbers
                        4. Conclusion
                        5. Provide depth

                        Context: {context}
                        Question: {question}""",
                        input_variables=["context", "question"]
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=False,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": prompt_template}
                    )

                    result = qa_chain({"query": prompt})
                    answer = result["result"].replace("- ", "• ").replace("* ", "• ")
                    container.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.info(TEMPLATES["welcome"])

if __name__ == "__main__":
    main()

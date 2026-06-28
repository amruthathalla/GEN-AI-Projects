import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key="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", 
    model_name="mixtral-8x7b-32768"  # or your preferred model
)
# -----------------------
# TITLE
# -----------------------

st.title("📄 PDF Chatbot")

# -----------------------
# CHAT HISTORY
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# LOAD GROQ KEY
# -----------------------

groq_key = st.secrets["GROQ_API_KEY"]

# -----------------------
# UPLOAD PDFS
# -----------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------
# PROCESS PDFS
# -----------------------

if uploaded_files:

    all_docs = []

    with st.spinner("Loading PDFs..."):

        for file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(file.getbuffer())

                loader = PyPDFLoader(tmp.name)

                docs = loader.load()

                all_docs.extend(docs)

    st.success("PDFs Loaded")

    # -----------------------
    # SPLIT TEXT
    # -----------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_docs)

    # -----------------------
    # EMBEDDINGS
    # -----------------------

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------
    # VECTOR DATABASE
    # -----------------------

    db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    # -----------------------
    # SHOW CHAT HISTORY
    # -----------------------

    for msg in st.session_state.messages:

        st.write(
            f"**{msg['role']}**"
        )

        st.write(
            msg["content"]
        )

    # -----------------------
    # QUESTION
    # -----------------------

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        # Search Similar Chunks

        docs = db.similarity_search(
            question,
            k=3
        )

        # Create Context

        context = ""

        for doc in docs:

            context += (
                doc.page_content
                + "\n\n"
            )

        # -----------------------
        # GROQ MODEL
        # -----------------------

        llm = ChatGroq(
             groq_api_key="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", 
                model_name="llama-3.3-70b-versatile"
        )

        # -----------------------
        # PROMPT
        # -----------------------

        prompt = f"""
        Answer only from the context below.

        If the answer is not found,
        say:

        I could not find that information in the document.

        Context:
        {context}

        Question:
        {question}
        """

        with st.spinner("Generating Answer..."):

            response = llm.invoke(prompt)

        answer = response.content

        # -----------------------
        # DISPLAY ANSWER
        # -----------------------

        st.subheader("Answer")

        st.write(answer)

        # -----------------------
        # SAVE CHAT
        # -----------------------

        st.session_state.messages.append(
            {
                "role": "User",
                "content": question
            }
        )

        st.session_state.messages.append(
            {
                "role": "Assistant",
                "content": answer
            }
        )

        # -----------------------
        # SOURCES
        # -----------------------

        with st.expander("View Sources"):

            for i, doc in enumerate(
                docs,
                start=1
            ):

                page = doc.metadata.get(
                    "page",
                    0
                )

                st.write(
                    f"Source {i}"
                )

                st.write(
                    f"📄 Page {page + 1}"
                )

                st.write(
                    doc.page_content[:300]
                )

                st.divider()
import streamlit as st
import os
import tempfile
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.memory import ConversationBufferMemory

# --- Fungsi Inti ---

def get_document_chunks(file_obj):
    """
    Reads a file (PDF or TXT), saves it temporarily, loads it,
    and splits it into chunks.
    """
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
            tmp_file.write(file_obj.getvalue())
            tmp_file_path = tmp_file.name

        st.info(f"File sementara dibuat di: {tmp_file_path}")

        # Load the temporary file based on its type
        if file_obj.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_obj.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        else:
            st.error("Format file tidak didukung. Harap unggah PDF atau TXT.")
            return None

        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    except Exception as e:
        st.error(f"Error saat memproses file: {e}")
        return None
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            st.info("File sementara dihapus.")

def get_or_create_vectorstore(chunks, openai_api_key, pinecone_api_key, index_name):
    """
    Initializes Pinecone, creates an index if it doesn't exist,
    and creates a PineconeVectorStore from the document chunks.
    """
    try:
        # Initialize OpenAI Embeddings
        # User requested "adasmall" (1536 dim), which is "text-embedding-ada-002"
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)

        # Check if index exists. If not, create it.
        if index_name not in pc.list_indexes().names():
            st.info(f"Membuat indeks baru: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimension for text-embedding-ada-002
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            st.success(f"Indeks {index_name} berhasil dibuat.")
        else:
            st.info(f"Menggunakan indeks yang sudah ada: {index_name}")

        # Create PineconeVectorStore
        # This will add the documents/chunks to the index
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
        )
        
        return vectorstore

    except Exception as e:
        st.error(f"Error saat inisialisasi Pinecone/Vectorstore: {e}")
        return None

def get_conversation_chain(vectorstore, openai_api_key):
    """
    Creates a conversational retrieval chain using the vectorstore.
    """
    try:
        # Initialize the LLM
        # User requested "model 5.1", which is invalid.
        # Using "gpt-3.5-turbo" as a robust default.
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            temperature=0.7
        )

        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Create the conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    except Exception as e:
        st.error(f"Error saat membuat conversation chain: {e}")
        return None

# --- UI Streamlit ---

st.set_page_config(page_title="Chat dengan Dokumen Anda", page_icon="💬")
st.title("Chatbot Tanya Jawab dengan Pinecone 💬")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# --- Sidebar ---
with st.sidebar:
    st.header("Pengaturan API 🔑")
    
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        help="Dapatkan di: https://platform.openai.com/account/api-keys"
    )
    pinecone_api_key = st.text_input(
        "Pinecone API Key", 
        type="password",
        help="Dapatkan di: https://app.pinecone.io/"
    )
    pinecone_index_name = st.text_input(
        "Pinecone Index Name",
        help="Nama untuk indeks Pinecone Anda (mis: 'chat-db')"
    )

    st.divider()

    st.header("Unggah Database Anda 📄")
    uploaded_file = st.file_uploader(
        "Unggah file PDF atau TXT Anda",
        type=["pdf", "txt"]
    )

    if st.button("Proses dan Setor ke Pinecone"):
        if not all([openai_api_key, pinecone_api_key, pinecone_index_name]):
            st.warning("Harap masukkan semua kunci API dan nama indeks.")
        elif not uploaded_file:
            st.warning("Harap unggah file terlebih dahulu.")
        else:
            with st.spinner("Membaca dan memecah dokumen..."):
                chunks = get_document_chunks(uploaded_file)
            
            if chunks:
                with st.spinner("Membuat embedding dan menyetor ke Pinecone... Ini mungkin perlu waktu."):
                    vectorstore = get_or_create_vectorstore(
                        chunks,
                        openai_api_key,
                        pinecone_api_key,
                        pinecone_index_name
                    )
                
                if vectorstore:
                    st.success("Dokumen berhasil diproses dan disimpan di Pinecone!")
                    st.session_state.processing_done = True
                    
                    # Create and store the conversation chain in session state
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore,
                        openai_api_key
                    )
                    st.info("Anda sekarang dapat mulai bertanya tentang dokumen Anda.")

# --- Area Chat Utama ---

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Terima input chat baru
if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    # Tambahkan pesan pengguna ke riwayat dan tampilkan
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cek apakah dokumen sudah diproses
    if not st.session_state.processing_done or st.session_state.conversation is None:
        st.warning("Harap unggah dan proses dokumen terlebih dahulu di sidebar.")
    else:
        # Hasilkan dan tampilkan respons AI
        with st.chat_message("assistant"):
            with st.spinner("Sedang berpikir..."):
                try:
                    # Panggil chain
                    response = st.session_state.conversation.invoke({"question": prompt})
                    response_text = response['answer']
                    
                    st.markdown(response_text)
                    # Tambahkan respons AI ke riwayat
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                except Exception as e:
                    st.error(f"Error saat menghasilkan respons: {e}")

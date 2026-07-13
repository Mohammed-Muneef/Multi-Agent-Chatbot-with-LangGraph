import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIR = "chroma_db"

def load_and_split_pdfs(folder_path):
    """Load PDFs from folder and split into chunks."""
    if not os.path.exists(folder_path):
        logger.warning(f"Folder {folder_path} does not exist. Creating it...")
        os.makedirs(folder_path, exist_ok=True)
        return []
    
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for file in pdf_files:
        try:
            file_path = os.path.join(folder_path, file)
            logger.info(f"Loading {file}...")
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            docs.extend(file_docs)
            logger.info(f"Loaded {len(file_docs)} pages from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    if not docs:
        logger.warning("No documents were loaded successfully")
        return []
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    split_docs = splitter.split_documents(docs) 
    return split_docs

def create_or_load_vectorstore(documents=None, name="docs"):
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # or "cuda" for GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"📄 Creating ChromaDB and embedding documents for {name}...")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=PERSIST_DIR, collection_name=name)
    vectordb.persist()
    return vectordb

def init_retrievers():
    invoice_docs = load_and_split_pdfs("invoices")
    order_docs = load_and_split_pdfs("shipping_orders")  # Fixed folder name to match downloaded dataset
    
    invoice_vectordb = create_or_load_vectorstore(invoice_docs, "invoices_pdfs")
    order_vectordb = create_or_load_vectorstore(order_docs, "orders_pdfs")
    
    orders_retriever = order_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    invoices_retriever = invoice_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    return orders_retriever, invoices_retriever

orders_retriever, invoices_retriever = init_retrievers()

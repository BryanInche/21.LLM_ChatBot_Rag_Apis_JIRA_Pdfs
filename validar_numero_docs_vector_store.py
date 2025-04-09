from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Cargar el vector store existente
vector_store3 = Chroma(
    persist_directory="5chroma_ms4m_bd_pdf_jira",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    collection_name="5data_ms4m_pdf_jira"
)

# Obtener el conteo exacto
total_documentos = vector_store3._collection.count()
print(f"Total de documentos/fragmentos almacenados: {total_documentos}")
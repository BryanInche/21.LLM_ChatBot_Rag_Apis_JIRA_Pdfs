"""
1. SCRIPT DE INTERACCIÓN CON EL CHATBOT RAG
========================================

Este módulo implementa la lógica central del sistema de preguntas y respuestas basado en:
1. Modelo LLM (Mistral a través de Ollama)
2. Base de datos vectorial (ChromaDB)
3. Técnicas RAG (Retrieval-Augmented Generation)

Funcionalidades clave:
- Carga y configuración del modelo de lenguaje
- Gestión del vector store con embeddings
- Sistema de recuperación de información semántica (RAG)
- Plantilla de prompt especializada para minería
- Memoria de conversación (opcional)
"""
from langchain_community.llms import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

#from cargar_solodata import cargar_documentos, crear_vectorstore
#from cargar_all_type_data import cargar_documentos, crear_vectorstore
from cargar_all_pdf_jira import obtener_datos_jira, jira_issues_a_documentos, cargar_documentos, crear_vectorstore

#from langchain_community.vectorstores import PGVector

from langchain.memory import ConversationBufferMemory # Agregar Memoria a tu Chatbot RAG

def iniciar_llm_chat(ruta_files):
    """
    # Inicializar el modelo de lenguaje (Mistral), Generar Texto
    Proceso detallado:
    1. Configuración del modelo LLM (Mistral via Ollama)
    2. Carga/creación del vector store con embeddings
    3. Configuración del sistema de recuperación semántica
    4. Creación de la plantilla de prompt especializada
    5. Ensamblaje de la cadena RAG completa
    """
    # 1. CONFIGURACIÓN DEL MODELO DE LENGUAJE (Mistral via Ollama)
    llm_ms4m = Ollama(model="mistral", # Modelo especializado en español
                      base_url="http://ollama:11434",  # Endpoint en Docker # Importante cuando quieres desplegar en Docker
                      temperature=0) #baja (o 0) para respuestas más deterministas 
    
    # 2. CONFIGURACIÓN DE EMBEDDINGS
    # Tarea de búsqueda semántica
    # Inicializar el modelo de embeddings (Hugging Face)
    # Modelo multilingüe. Esto significa que puede generar embeddings para textos en inglés y español
    embeding_modelo = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. GESTIÓN DEL VECTOR STORE
    # Intentar cargar el vector store existente
    try:
        vs_ms4m = Chroma(
            embedding_function=embeding_modelo,
            persist_directory="5chroma_ms4m_bd_pdf_jira",
            collection_name="5data_ms4m_pdf_jira"
        )
    except Exception as e:
        print(f"No se pudo cargar el vector store existente: {e}")
        print("Creando un nuevo vector store...")
        # Cargar documentos y crear el vector store
        #documentos = cargar_documentos(ruta_files)

        ## Carga Solo API de JIRA
        config_jira = {
            'jql': 'project = MS4M AND resolutiondate >= "2024-01-01" AND resolutiondate <= "2025-03-31" AND status = "CLOSED" ',
            'fields': ["key", "summary", "status", "resolution", "resolutiondate", "description", "comment", "customfield_11281",
            "customfield_11228", "customfield_10975"]
        }
        
        documentos = cargar_documentos(rutas_archivos=ruta_files,jira_config=config_jira)
        
        vs_ms4m = crear_vectorstore(documentos)
    
    # 4. CONFIGURACIÓN DEL RETRIEVER
    # Crear el retriever para buscar en el vector store
    #- Similarity (Similitud Coseno) : Calcula el ángulo entre estos vectores: menor ángulo = mayor similitud
    #- MMR (Maximum Marginal Relevance) : Similitud con la pregunta y Diversidad entre los documentos seleccionados
    #- Similarity Score Threshold : Primero calcula la similitud coseno normal, Luego filtra por un umbral mínimo
    retriver_ms4m = vs_ms4m.as_retriever(
        #search_type="similarity",  # Búsqueda por similitud coseno
        search_kwargs={"k": 3})     # Número de documentos a recuperar
    
    # 5. Definir el prompt personalizado para Responder al usuario
    prompt_template_ms4m = """
    Eres un experto ingeniero de minas de minería subterránea y tajo abierto. Tu tarea es responder preguntas utilizando únicamente la información proporcionada en el contexto. 
    Sigue las siguientes instrucciones:

    1. Si la pregunta está en español y el contexto que obtienes está en inglés, traduce la información del contexto al español y luego genera la respuesta en español.
    2. Si no sabes la respuesta o la información no está en el contexto, responde: "No tengo información suficiente para responder a esa pregunta". No alucines porfavor.
    3. No inventes respuestas. Solo utiliza la información proporcionada en el contexto.

    Contexto: {context}
    Pregunta: {question}

    Responde únicamente en español y de manera profesional, concisa y precisa.
    Respuesta útil:
    """

    prompt_1 = PromptTemplate(
        template=prompt_template_ms4m,
        input_variables=["context", "question"]
    )

    # # 6. CONFIGURACIÓN DE LA CADENA RAG(RetrievalQA)
    cadena_rag = RetrievalQA.from_chain_type(
        llm=llm_ms4m,                      # Modelo de lenguaje (Mistral)
        chain_type="stuff",                # Tipo de cadena ("stuff"),Método para manejar múltiples documentos
        retriever=retriver_ms4m,           # Retriever para buscar documentos
        return_source_documents=True,      # Devuelve los documentos fuente
        chain_type_kwargs={"prompt": prompt_1}  # Prompt personalizado
    )

    
    return cadena_rag

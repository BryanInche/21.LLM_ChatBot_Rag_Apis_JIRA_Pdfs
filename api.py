"""
1.BACKEND PARA CHATBOT LLM RAG
============================

Este módulo implementa una API REST usando FastAPI que sirve como backend para un chatbot
con capacidades RAG (Retrieval-Augmented Generation). Se conecta con un modelo LLM (Mistral)
a través de Ollama y permite realizar búsquedas en documentos corporativos.

Características principales:
- Procesamiento de preguntas en lenguaje natural
- Búsqueda semántica en documentos (PDF, Word, PPT)
- Generación de respuestas contextualizadas
- Configuración CORS para integración con frontend
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Importa el middleware CORS
from pydantic import BaseModel
from chat2_llm import iniciar_llm_chat  # Importa tu función principal

# Inicializa la aplicación FastAPI
app = FastAPI()

"""
2.Configuración CORS (Cross-Origin Resource Sharing)
-------------------------------------------------
Habilita la comunicación segura entre el frontend (React) y el backend (FastAPI)
en diferentes escenarios:
- Desarrollo local (localhost)
- Entorno Docker (frontend service)
- Acceso desde otras máquinas en la red local
"""
app.add_middleware( #orígenes que haran peticiones al Backend
    CORSMiddleware,
    allow_origins=["http://192.168.5.163:3000", # IP del servidor donde corre el frontend
                   "http://localhost:3000",  # Desarrollo local
                   "http://frontend:3000"],  # Nombre del servicio en Docker
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

"""
3. Define el modelo de entrada para la API,
-----------------------------------
Define la estructura esperada para las preguntas entrantes.
Usamos Pydantic para validación automática de datos.
"""
class QueryRequest(BaseModel):
    question: str

"""
4. Inicialización del Sistema RAG
------------------------------
Carga los documentos y prepara el sistema de búsqueda semántica(EN CASO NO SE CARGUE EL VECTOR STORE ACTUAL)
Se ejecuta una sola vez al iniciar la aplicación.
"""
#ruta_files = [
#    #"pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",  # Ruta a un archivo PDF
#    "ppt_solos",  # Ruta a una carpeta de PPTs
#    "word_solos",  # Ruta a una carpeta de Words
#    "pdf_carpetas/pdfs_varios"  # Ruta a una carpeta de PDFs
#]

## 1.Carga solo archivos locales 
ruta_files = [
    #"pdf_solos/ASIS_REPORTE_EJECUTIVO.pdf",  # Ruta a un archivo PDF
    "ppt_solos",  # Ruta a una carpeta de PPTs
    "word_solos",  # Ruta a una carpeta de Words
    "pdf_carpetas/pdfs_varios"  # Ruta a una carpeta de PDFs
#    "json_solos"
]

cadena_rag = iniciar_llm_chat(ruta_files)

"""
5. Endpoint Principal: /respuesta_llm
---------------------------------
Procesa las preguntas del usuario y devuelve respuestas contextualizadas
con referencias a los documentos fuente.

Flujo de procesamiento:
1. Recibe pregunta en formato JSON
2. Ejecuta búsqueda semántica en los documentos
3. Genera respuesta usando el modelo LLM
4. Devuelve respuesta + documentos fuente
"""
@app.post("/respuesta_llm")
def ask_question(query: QueryRequest):
    try:
        # # Procesamiento de la pregunta
        respuesta = cadena_rag.invoke({"query": query.question})
        # Formateo de la respuesta
        return {
            "response": respuesta["result"], # Respuesta generada 
            "source_documents": [            # Documentos de referencia
                {
                    #"page": doc.metadata["page"], # Usa get() para evitar KeyError 
                    #"source": doc.metadata["source"] # Ruta del documento
                    "source": doc.metadata.get("source", "Sin fuente"),
                    "page": doc.metadata.get("page", "N/A")  # None por defecto
                }
                for doc in respuesta["source_documents"]
            ]
        }
    except Exception as e:
        # Manejo estructurado de errores
        raise HTTPException(status_code=500, detail=str(e))
    
    

"""
6. Punto de Entrada Principal que Ejecuta la API con Uvicorn
--------------------------
Configuración del servidor Uvicorn para producción.
"""
if __name__ == "__main__":
    #host="127.0.0.1" → La API solo es accesible desde la misma máquina (localhost).
    #host="0.0.0.0" → La API es accesible desde cualquier dispositivo en la red: Util cuando : 
    # 1.Quieres acceder a la API desde otro equipo en la misma red
    uvicorn.run(app, 
                host="0.0.0.0", # Escucha en todas las interfaces de red
                port=8000)      # Puerto de escuchara el backend
    

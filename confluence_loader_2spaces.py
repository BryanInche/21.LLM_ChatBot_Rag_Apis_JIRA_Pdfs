# confluence_loader.py
import os
import re
import logging
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin
from markdownify import markdownify as md
from langchain_core.documents import Document as LangchainDocument

class ConfluenceLoader:
    def __init__(self, email, api_token, base_url, space_keys, export_dir):
        self.email = email
        self.api_token = api_token
        self.base_url = base_url
        self.space_keys = space_keys if isinstance(space_keys, list) else [space_keys] # Aceptamos 2 o mas spaces of confluence
        self.export_dir = export_dir
        self.headers = {"Accept": "application/json"}
        self.auth = HTTPBasicAuth(email, api_token)
        
        os.makedirs(self.export_dir, exist_ok=True)

    def get_top_level_pages(self, space_key):
        """Obtiene páginas raíz de un espacio específico de Confluence"""
        url = f"{self.base_url}/rest/api/space/{space_key}/content" # Se usara space_key, por cada uno
        params = {"expand": "body.storage", "limit": 50}
        
        try:
            response = requests.get(url, headers=self.headers, auth=self.auth, params=params)
            #response.raise_for_status()
            data = response.json()
            return data.get("results") or data.get("page", {}).get("results", [])
        except Exception as e:
            logging.error(f"Error al obtener páginas del espacio {space_key}: {str(e)}")
            return []

    def clean_filename(self, title):
        """Limpia títulos para usarlos en rutas de archivos"""
        return re.sub(r'[<>:"/\\|?*]', '_', title).strip()

    def save_page_with_images(self, page, space_key):
        """Guarda una página como Markdown con sus adjuntos"""
        try:
            title = self.clean_filename(page["title"])
            page_id = page["id"]
            # Incluimos el space_key en la ruta para evitar conflictos
            page_dir = os.path.join(self.export_dir, f"{space_key}_{title}")
            os.makedirs(page_dir, exist_ok=True)

            html_content = page["body"]["storage"]["value"]
            markdown_content = md(html_content)

            with open(os.path.join(page_dir, "page.md"), "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{markdown_content}")

            logging.info(f"Página exportada: {title} (Espacio: {space_key})")
            return True
        except Exception as e:
            logging.error(f"Error al exportar '{page.get('title')}' del espacio {space_key}: {str(e)}")
            return False

    def load_confluence_documents(self):
        """Carga documentos desde múltiples espacios de Confluence y los convierte a formato LangChain"""
        all_documents = []
        
        for space_key in self.space_keys:
            logging.info(f"Procesando espacio: {space_key}")
            pages = self.get_top_level_pages(space_key)
            if not pages:
                logging.warning(f"No se encontraron páginas en el espacio {space_key}.")
                continue

            # Exportar páginas a Markdown, para guardar los documentos de todos los Espacios generados
            for page in pages:
                self.save_page_with_images(page, space_key)

        # Cargar todos los documentos Markdown como objetos LangChain
        for folder_name in os.listdir(self.export_dir):
            page_dir = os.path.join(self.export_dir, folder_name)
            md_file = os.path.join(page_dir, "page.md")

            if os.path.exists(md_file):
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extraemos el space_key del nombre de la carpeta si es necesario
                    space_key = folder_name.split("_")[0]
                    all_documents.append(
                        LangchainDocument(
                            page_content=content,
                            metadata={
                                "source": "confluence",
                                "title": "_".join(folder_name.split("_")[1:]),
                                "space_key": space_key,
                                "file_path": md_file
                            }
                        )
                    )

        return all_documents
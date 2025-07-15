# Sistema de Preguntas y Respuestas con RAG (Retrieval-Augmented Generation)

Este proyecto implementa un sistema de preguntas y respuestas usando RAG con LangChain. Permite la carga de documentos, su vectorización y la consulta semántica mediante modelos LLM (OpenAI o modelo local).

## 🚀 Tecnologías utilizadas

- FastAPI: Backend para la API REST.
- Streamlit: Frontend interactivo para usuarios.
- LangChain: Para la orquestación de cadenas RAG.
- ChromaDB: Base vectorial para almacenamiento y recuperación semántica.
- OpenAI: Modelos GPT-4o, GPT-4o-mini.
- Microsoft Phi-4 Mini: Modelo local para ejecución privada.
- HuggingFace Transformers: Para cargar modelos LLM y tokenizadores.
- Sentence Transformers: Para embeddings multilingües.

## 🧩 Características

- Ingesta de documentos (.csv, .xlsx) | Posibilidad de agregar docx, pdf a futuro
- Consulta semántica con RAG (Retriever + LLM)
- Selección entre modelo local o OpenAI desde el frontend
- Simulación de autenticación básica de usuario
- Historial de sesiones por usuario

## 📁 Estructura del proyecto
```bash
.
├── app/                  # Frontend (Streamlit)
│   ├── app_front.py
│   ├── sidebar.py
│   └── login.py
├── api/                  # Backend y lógica de negocio
│   ├── main.py           # Endpoints FastAPI
│   ├── langchain_utils.py
│   ├── db_utils.py       # Autenticación y logs
│   ├── chroma_utils.py   # Vector store
│   └── api_utils.py      # Cliente HTTP
├── documents/            # Archivos cargados
├── embeddings/           # Vector store local
├── .env                  # Claves y configuración
├── requirements.txt
└── README.md
```

## ⚙️ Instalación y ejecución local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r requirements.txt

# Variables de entorno
cp .env.example .env
# Edita .env y agrega tu OPENAI_API_KEY si usarás OpenAI

# Iniciar backend (API)
uvicorn api.main:app --reload

# Iniciar frontend
streamlit run app/app_front.py
```

## 🔐 Requisitos del modelo local (Phi-4 Mini)

- RAM: → mínimo 8GB (ideal 16GB)
- GPU: → NVIDIA con al menos 4GB VRAM (opcional, pero mejora rendimiento)
- Dependencias: transformers, torch, accelerate
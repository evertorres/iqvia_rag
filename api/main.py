from fastapi import FastAPI, File, UploadFile, HTTPException
from .pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from .langchain_utils import get_rag_chain
from .db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from .chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
import shutil
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi import Form
from .db_utils import get_user_by_username, hash_password

load_dotenv()

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()


# Habilitar CORS por si se usa desde otro puerto
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API funcionando"}

### Ask

@app.post("/ask", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    
    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    
    result = rag_chain.invoke({
                                    "input": query_input.question,
                                    "chat_history": chat_history
                              })

    answer = result['answer']
    context_items = result.get('context', [])
    context_texts = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in context_items]
    
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    
    return QueryResponse(
            answer=answer,
            session_id=session_id,
            model=query_input.model,
            context=context_texts)


#### POST UPLOAD FILE ####

@app.post("/upload")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.csv', '.xlsx']
    file_extension = os.path.splitext(file.filename)[1].lower()
    print(file_extension)

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
    

@app.post("/login")
def login_user(username: str = Form(...), password: str = Form(...)):
    user = get_user_by_username(username)
    if user:
        if user["password"] == hash_password(password):
            return {"authenticated": True}
        return {"authenticated": False}

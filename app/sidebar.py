import streamlit as st
from api_utils import upload_document, list_documents, delete_document

def display_sidebar():
    # Sidebar: Model Type Selection
    model_type = st.sidebar.radio(
        "Select the LLM Provider",
        options=["OpenAI", "Local"],
        help="Select 'Local' to use Phi-4-mini (Microsoft) locally"
    )
    
    st.session_state["model_type"] = model_type.lower()

    # Sidebar: Model Selection based on type
    if model_type == "Local":
        model_options = {"Phi-4 Mini (Microsoft)": "local"}
    else:
        model_options = {
            "GPT-4o": "gpt-4o",
            "GPT-4o Mini": "gpt-4o-mini"
        }

    selected_label = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        key="model" 
    )

    selected_model_id = model_options[selected_label]

    if selected_model_id == "local":
        st.warning(
            "El modelo local Phi-4 Mini requiere al menos:\n\n"
            "- 8 GB de RAM física (idealmente 16 GB)\n"
            "- 6+ GB de memoria GPU (NVIDIA)\n"
            "- PyTorch y Transformers instalados correctamente\n\n"
            "Si no se cumplen estos requisitos, el modelo puede ejecutarse lentamente o fallar.",
            icon="⚠️"
        )

    st.session_state["selected_model_id"] = selected_model_id.lower()
    print('Modelo', st.session_state["model"])

    # Sidebar: Upload Document
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully with ID {upload_response['file_id']}.")
                    st.session_state.documents = list_documents()  # Refresh the list after upload

    # Sidebar: List Documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()

    # Initialize document list if not present
    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})")
        
        # Delete Document
        selected_file_id = st.sidebar.selectbox("Select a document to delete", options=[doc['id'] for doc in documents], format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x))
        if st.sidebar.button("Delete Selected Document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully.")
                    st.session_state.documents = list_documents()  # Refresh the list after deletion
                else:
                    st.sidebar.error(f"Failed to delete document with ID {selected_file_id}.")
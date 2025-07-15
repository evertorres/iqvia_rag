from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from .chroma_utils import vectorstore
from dotenv import load_dotenv

load_dotenv()


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()



# Prompts Para OpenAI
contextualize_q_system_prompt = (
    "Dado un historial de conversación y la pregunta más reciente del usuario,"
    "reformula la pregunta para que sea autónoma y se entienda sin necesidad del contexto previo."
    "NO respondas la pregunta, solo reformúlala si es necesario. De lo contrario, devuélvela tal como está."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de inteligencia artificial útil y preciso. Usa el siguiente contexto para responder la pregunta del usuario."),
    ("system", "Contexto: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

#Construir Prompt plano con historia para local
def build_plain_prompt_with_history(context: str, question: str, chat_history: list[dict]) -> str:
    history_str = ""
    for msg in chat_history:
        if msg["role"] == "user":
            history_str += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            history_str += f"<|assistant|>\n{msg['content']}\n"

    return (
        f"<|system|>\nEres un asistente útil y preciso. Usa el contexto para responder la pregunta.\n"
        f"<|context|>\n{context}\n"
        f"{history_str}"
        f"<|user|>\n{question}\n"
        f"<|assistant|>"
    )

phi_model = None

def load_phi4_llm(model_name="microsoft/Phi-4-mini-instruct"):
    global phi_model #Global para evitar recarga.
    
    if phi_model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=False
        )
        phi_model = HuggingFacePipeline(pipeline=pipe)
    return phi_model 

def get_rag_chain(model="gpt-4o-mini"):
    if model in ["local", "phi4", "phi-4", "microsoft/Phi-4-mini-instruct"]:
        llm = load_phi4_llm()

        def rag_pipeline(inputs: dict) -> dict:
            question = inputs["input"]
            chat_history = inputs.get("chat_history", [])
            docs: list[Document] = retriever.get_relevant_documents(question)
            context = "\n".join(doc.page_content for doc in docs)
            prompt_text = build_plain_prompt_with_history(context, question, chat_history)
            answer = llm.invoke(prompt_text)
            return {
                "answer": answer,
                "context": [doc.page_content for doc in docs]
            }

        return RunnableLambda(rag_pipeline)

    else:
        llm = ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain
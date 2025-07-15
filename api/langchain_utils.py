from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from .chroma_utils import vectorstore
from dotenv import load_dotenv

load_dotenv()


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()



# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

qa_prompt_plain = PromptTemplate.from_template(
    "<|system|>\nYou are a helpful AI assistant.\n"
    "<|context|>\n{context}\n"
    "<|user|>\n{input}\n"
    "<|assistant|>")

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
        prompt = qa_prompt_plain
    else:
        llm = ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        prompt = qa_prompt

    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
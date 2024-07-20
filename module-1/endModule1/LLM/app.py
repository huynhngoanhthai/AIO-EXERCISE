"""Xây dựng giao diện chat Chainlit."""

import chainlit as cl
import torch
from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer , AutoModelForCausalLM , pipeline
from langchain_huggingface . llms import HuggingFacePipeline

from langchain . memory import ConversationBufferMemory
from langchain_community . chat_message_histories import ChatMessageHistory

from langchain . chains import ConversationalRetrievalChain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community . document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core . runnables import RunnablePassthrough
from langchain_core . output_parsers import StrOutputParser
from langchain import hub

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

embedding = HuggingFaceEmbeddings()

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    else:
        raise ValueError(f"Unsupported file type: {file.type}")

    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)  # Ensure text_splitter is defined/imported
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs

def get_vector_db(file: AskFileResponse):
    docs = process_file(file)  # Assuming process_file is defined elsewhere
    cl.user_session.set("docs", docs)  # Assuming cl and user_session are defined elsewhere

    # Assuming embedding is defined or imported from HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    return vector_db

def get_huggingface_llm(model_name: str = "lmsys/vicuna-7b-v1.5", max_new_token: int = 512):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )

    return llm

LLM = get_huggingface_llm()

welcome_message = """Welcome to the PDF QA! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing ‘{file.name}’...", disable_feedback=True)
    await msg.send()

    vector_db = await cl.make_async(get_vector_db)(file)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"‘{file.name}’ processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])

    answer = res.get("answer", "")
    source_documents = res.get("source_documents", [])
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

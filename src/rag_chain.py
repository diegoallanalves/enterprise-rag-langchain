from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import CHROMA_DIR, PROMPT_FILE, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from data_quality_checks import build_quality_summary

def get_retriever():
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

def run_rag_query(question):
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([f"Source: {d.metadata}\n{d.page_content}" for d in docs])
    quality_summary, quality_df = build_quality_summary()
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nRetrieved context:\n{context}\n\nData quality checks:\n{quality_summary}")
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    response = (prompt | llm).invoke({
        "question": question,
        "context": context,
        "quality_summary": quality_summary
    })

    return response.content, docs, quality_df

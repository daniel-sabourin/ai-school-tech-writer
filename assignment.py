import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAI

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate


def main():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(api_key=key, model="gpt-3.5-turbo-0125")

    # text = "What is a creative name an escape room business?"
    # system_text = "All of your answers must start with the same letter and be at least 5 words."

    # messages = [SystemMessage(content=system_text), HumanMessage(content=text)]

    # response_llm = llm.invoke(messages)
    # print(response_llm)
    # print("*******")


    embeddings = OpenAIEmbeddings(api_key=key, model="text-embedding-3-small")

    # query it
    query = "What are some challenges with building LLM agents?"

    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever()
    context = retriever.get_relevant_documents(query)
    for doc in context:
        print(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n")
    print("__________________________")

    # Adding context to our prompt
    template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
    prompt_with_context = template.invoke({"query": query, "context": context})

    # Asking the LLM for a response from our prompt with the provided context
    llm = ChatOpenAI(temperature=0.7)
    results = llm.invoke(prompt_with_context)

    print(results.content)

if __name__ == '__main__':
    main()

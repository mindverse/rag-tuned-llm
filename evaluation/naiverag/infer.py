import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

import json

# basic setting
os.environ["COHERE_API_KEY"] = ''

# llm and embedding setting
llm = ChatOpenAI(model="gpt-4o-mini", api_key="")
embedding=OpenAIEmbeddings(api_key="")

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# def no_rerank(file_path, llm, embedding, text_splitter, prompt, query, top_k1=10):
#     docs = TextLoader(file_path, encoding='utf-8').load()
#     splits = text_splitter.split_documents(docs)
#     vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": top_k1})
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     chain = create_retrieval_chain(retriever, question_answer_chain)
#     result = chain.invoke({"input": query})
#     return result

def with_rerank(file_path, llm, embedding, text_splitter, prompt, queries, top_k1=10, top_k2=5):
    documents = TextLoader(file_path).load()
    texts = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(
        texts, embedding
    ).as_retriever(search_kwargs={"k": top_k1})
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=top_k2)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    results = []
    for idx, query in enumerate(queries):
        tmp_res = chain.invoke({"input": query})
        results.append(tmp_res['answer'])
        print(f"Finished query {idx+1}/{len(queries)}")
    return results

# # podcast dataset
# txt_path = '/mnt/datadisk0/wjl/podcast/podcast_data.txt'
# questions_path = '/mnt/datadisk0/wjl/research_paper/questions_podcast.json'
# questions = json.load(open(questions_path))
# queries = [q['question'] for q in questions]
# podcast_results = with_rerank(txt_path, llm, embedding, text_splitter, prompt, queries)
# # save results in the format of [{"question": xxx, "answer": xxx}, ...] pairs
# podcast_results = [{"question": q, "answer": r} for q, r in zip(queries, podcast_results)]
# json.dump(podcast_results, open('/mnt/datadisk0/wjl/research_paper/podcast_naiverag_results_101216.json', 'w'), indent=4)

# # news dataset
# txt_path = '/mnt/datadisk0/wjl/news/news_data.txt'
# questions_path = '/mnt/datadisk0/wjl/research_paper/questions_news.json'
# questions = json.load(open(questions_path))
# queries = [q['question'] for q in questions]
# news_results = with_rerank(txt_path, llm, embedding, text_splitter, prompt, queries)
# news_results = [{"question": q, "answer": r} for q, r in zip(queries, news_results)]
# json.dump(news_results, open('/mnt/datadisk0/wjl/research_paper/news_naiverag_results_101216.json', 'w'), indent=4)

# lpm dataset
txt_path = '/mnt/datadisk0/wjl/naiverag/merge.txt'
questions_path = '/mnt/datadisk0/wjl/research_paper/lpm_local_search_c0_full.json'
questions = json.load(open(questions_path))
queries = [q['question'] for q in questions]
news_results = with_rerank(txt_path, llm, embedding, text_splitter, prompt, queries)
news_results = [{"question": q, "answer": r} for q, r in zip(queries, news_results)]
json.dump(news_results, open('/mnt/datadisk0/wjl/research_paper/0924_lpm/lpm_naiverag_results_1014.json', 'w'), indent=4, ensure_ascii=False)

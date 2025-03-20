import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

txt_path = './podcast.txt'
os.environ["COHERE_API_KEY"] = ''

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

openai_embeddings = OpenAIEmbeddings(api_key="", base_url="")

documents = TextLoader(txt_path).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(
    texts, openai_embeddings
).as_retriever(search_kwargs={"k": 10})

query = "Who is the host of Behind the Tech?"
# docs = retriever.invoke(query)
# pretty_print_docs(docs)

llm = ChatOpenAI(model="", api_key="", base_url="")
compressor = CohereRerank(model="rerank-english-v3.0", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
# compressed_docs = compression_retriever.invoke(query)
# pretty_print_docs(compressed_docs)
from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(compression_retriever, question_answer_chain)

result = chain.invoke({"input": query})
print(len(result['context']))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.memory import ConversationBufferMemory

def load_pdf(num_page):
    """
    This function loads the book from local directory and loads {num_page} pages for indexing

    Args:
        num_page (int): number of page to pass on to index

    Returns:
        list: list of [Document]
    """
    loader = PyPDFium2Loader("./books/book.pdf")
    docs_before_split = loader.load()
    docs_selected_page = []
    for i in  range(len(docs_before_split)+1):
        if docs_before_split[i].metadata['page'] <= int(num_page)+13:
            docs_selected_page.append(docs_before_split[i])
        else:
            break
    return docs_selected_page

def text_chunk(text):
    """
    This function recursively tries to split by different characters to find one that works

    Args:
        text (list): Output of load_pdf

    Returns:
        list: list of [Document]
    """
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],
    chunk_size = 700,
    chunk_overlap  = 50,
    is_separator_regex=True)
    docs_after_split = text_splitter.split_documents(text)
    return docs_after_split

def vector_store(text):
    """
    This function creates embeddings for chunks and stores in memory

    Args:
    text (list): Output of text_chunk

    Returns:
        None

    """
    huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2",
                                                  model_kwargs={'device':'cpu'}, 
                                                  encode_kwargs={'normalize_embeddings': True})
    vectorstore1 = FAISS.from_documents(text, huggingface_embeddings)
    vectorstore1.save_local("faiss_index2")

def get_retrieval_qa():
    prompt_template = """
    Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer".
    2. Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    PROMPT = PromptTemplate(
     template=prompt_template, input_variables=["context", "question"])
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                          model_kwargs={'device':'cpu'}, 
                                          encode_kwargs={'normalize_embeddings': True})

    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.5)
    load_vector = FAISS.load_local("faiss_index2",huggingface_embeddings,allow_dangerous_deserialization=True)
    retriever = load_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"verbose":True, "prompt": PROMPT,"memory": ConversationBufferMemory(memory_key="history",
            input_key="question")})
    return retrievalQA
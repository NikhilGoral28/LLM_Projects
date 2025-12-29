from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import  PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA


#loading api
load_dotenv()



#creating llm

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.environ["GOOGLE_api_key"],
    temperature=0.7
)

#loading the csv
loader = CSVLoader(file_path=r'D:\LLM_Projects\LLM_deoms\final_movies.csv',encoding='utf-8', source_column='tags')

data = loader.load()

#embedding
instructor_embedding = HuggingFaceInstructEmbeddings()

vector_db_file_path = "faiss_index"

def create_vector_db ():
    #loading the csv
    loader = CSVLoader(file_path=r'D:\LLM_Projects\LLM_deoms\final_movies.csv',encoding='utf-8', source_column='tags')

    data = loader.load()

    vectordb = FAISS.from_documents(documents=data,embedding=instructor_embedding)

    #save to local
    vectordb.save_local(vector_db_file_path)


def get_qa_chain():

    #load the vector databse from local file
    vectordb = FAISS.load_local(vector_db_file_path, instructor_embedding)

    #create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """ Given the following context and a question, generate an answer based on this context. In the answer try to provide as much text as possibel from "tags and title section in the source document . If the answer not found in the context, kindaly state " I dont know" Dont try to make up an answer.
    CONTEXT: {context}

    QUESTION: {question} """

    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )

    #creating a chain
    chain = RetrievalQA(llm=llm,
            chain_type = "stuff",
            retriever = retriever,
            input_key = "query",
            return_source_documents = True,
            chain_type_kwargs = {"prompt": PROMPT}
    )

    return chain



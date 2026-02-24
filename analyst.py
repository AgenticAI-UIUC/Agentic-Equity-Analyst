from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
import os

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

filings = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="company_filings", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

parser = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="parser_data", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

news = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="news_articles", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

financials = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="financial_data", 
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)

model = init_chat_model("gpt-4o", model_provider="openai")


def analyze(query, collection):
    try:
        res = collection.similarity_search(query=query, k=10)

        messages = [SystemMessage(content="You are a professional technical financial analyst."),
                    HumanMessage(content=f"Summarize the following data: {res[:]} . Do not repeat yourself"),
                    ]
        
        return model.invoke(messages).content
    except Exception as e:
        return f"Error while running tool: {e}"

@tool
def analyze_filings(query):
    """
    Fetches info relevant to what you would find in a 10-K or 10-Q filing of a company using the data fetched in the database
    Takes just one string as an argument 
    """
    return analyze(query, filings)

@tool
def analyze_news(query):
    """
    Fetches info relevant to what you would find in the news about a company using the data fetched in the database
    Takes just one string as an argument 
    """
    return analyze(query, news)

@tool
def analyze_parser(query):
    """
    Fetches info relevant to what you would find in the news about a company using the data fetched in the database
    Takes just one string as an argument 
    """
    return analyze(query, parser)

@tool
def analyze_financials(query):
    """
    Fetches info about a company's ticker data  
    Takes just one string as an argument 
    """
    return analyze(query, financials)

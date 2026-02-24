import uuid, os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from langchain.tools import tool 

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


EMBEDDING_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
parser_data = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="parser_data",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)
company_filings = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="company_filings",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)
yfinance_data = Chroma(
    database=os.getenv("CHROMADB"),
    collection_name="financial_data",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
)
model = init_chat_model("gpt-4o", model_provider="openai")
txt_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.35)
TXT_THRESHOLD = 200

def chunk_text(txt):
  if (len(txt) > TXT_THRESHOLD):
    return txt_splitter.split_text(txt)
  
  return [txt]
  
def clean_text_list(texts):
  cleaned = []

  for t in texts:
    if not isinstance(t, str):
      t = str(t)

      t = re.sub(r'\s+', ' ', t).strip()
      cleaned.append(t)

  return cleaned

def clean_text(text):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def parse(company: str, year: str):
  with open("parser_queries.txt", "r") as f:
    for line in f:
      line = line.strip()
      spl = line.split(":")
      query = company + " " + spl[0] + " " + "in " + year

      if spl[1] == "yf":
        res = yfinance_data.similarity_search(query=query, k=10)
      else:
        res = company_filings.similarity_search(query=query, k=10)

      cleaned_text = clean_text(" ".join([doc.page_content for doc in res]) if isinstance(res, list) else str(res))
      messages = [
        SystemMessage(content="You are a professional financial equity analyst. Always produce clean, continuous text without unnecessary line breaks or bullet points."), 
        HumanMessage(content=f"Summarize and analyze the following data. Write in paragraph form only (no line breaks, lists, or formatting): {cleaned_text}")
      ]
      txt = chunk_text(model.invoke(messages).content)
      ids = [f"{company}_{year}_{uuid.uuid4()}_{i}" for i in range(len(txt))]
      txt = clean_text_list(txt)
      parser_data.add_texts(texts=txt,metadatas=[{"company": company,"year": year} for _ in txt],ids=ids)

@tool
def parse_tool(company: str, year: str):
  """
  Takes in two strings as its arguments: the company name, and the year (formatted as XXXX)
  """
  return parse(company, year)

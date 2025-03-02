from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, select, text
from fastapi import FastAPI, HTTPException, UploadFile, File, status, Depends
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError, ExpiredSignatureError
from sqlalchemy.ext.declarative import declarative_base
from langchain_openai import AzureOpenAIEmbeddings
from sqlalchemy.dialects.postgresql import JSONB, UUID
from jose.exceptions import JWTClaimsError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
import numpy as np
import requests
import random
import logging
import string
import json
import hashlib
import uuid
import io

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# CORS middleware settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:******",
        "https://localhost:******",
        "https://192.168.0.***:******",
        "https://llm.platform.******",
        "https://rossana.platform.******"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS", "HEAD"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["*"],
    max_age=3600
)

# Error handling middleware
@app.middleware("http")
async def add_error_handling(request: Request, call_next):
    try:
        logger.debug(f"Incoming request: {request.method} {request.url}")
        logger.debug(f"Headers: {request.headers}")

        response = await call_next(request)

        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

embeddings = OllamaEmbeddings(model='******', base_url='http://******:11434')

# Database connection
CONNECTION_STRING = "postgresql+psycopg2://******:******@localhost:5432/******"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Auth0 Configuration
AUTH0_CONFIG = {
    "Domain": "******.auth0.com",
    "Audience": "******",  
    "Issuer": "https://******.auth0.com/",
}

ALGORITHMS = ["RS256"]

# Fetch Auth0 public key
def get_auth0_public_key():
    url = f"https://{AUTH0_CONFIG['Domain']}/.well-known/jwks.json"
    response = requests.get(url)
    return response.json()

# File processing functions
def process_pdf(file):
    with open("temp_pdf.pdf", "wb") as f:
        f.write(file)
    loader = PyPDFLoader("temp_pdf.pdf")
    return loader.load()

def process_docx(file):
    with open("temp_docx.docx", "wb") as f:
        f.write(file)
    loader = UnstructuredWordDocumentLoader("temp_docx.docx")
    return loader.load()

def process_txt(file):
    with open("temp_txt.txt", "wb") as f:
        f.write(file)
    loader = TextLoader("temp_txt.txt")
    return loader.load()

def process_csv(file):
    with open("temp_csv.csv", "wb") as f:
        f.write(file)
    loader = CSVLoader("temp_csv.csv")
    return loader.load()

def process_xlsx(file):
    with open("temp_xlsx.xlsx", "wb") as f:
        f.write(file)
    loader = UnstructuredExcelLoader("temp_xlsx.xlsx")
    return loader.load()

# Text splitting and embedding functions
def split_text(docs, chunk_size=3000, overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))
    return chunks

def create_embeddings(text_chunks, embeddings):
    return [embeddings.embed_query(chunk) for chunk in text_chunks]

# Database model
Base = declarative_base()

class LangchainPgEmbedding(Base):
    __tablename__ = 'langchain_pg_embedding'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pageContent = Column(String, name='pageContent')
    file_metadata = Column(JSONB, name='metadata')
    embedding = Column(Vector, name='embedding')
    equipment_id = Column(Integer, name='equipment_id')
    document_id = Column(Integer, name='document_id')

# Database connection setup
engine = create_engine(CONNECTION_STRING)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# RBAC Functions
def get_user_role(token_payload):
    roles = token_payload.get('https://******.com/roles', [])
    tenants = token_payload.get('https://******.com/mywai_tenants', [])
    
    if "admin" in roles:
        return "admin"
    elif roles or tenants:
        return "user"
    return "guest"

def has_permission(role, required_role):
    role_hierarchy = {"admin": 3, "user": 2, "guest": 1}
    return role_hierarchy.get(role, 0) >= role_hierarchy.get(required_role, 0)

@app.post("/generate_embeddings", response_model=BaseModel)
async def generate_embeddings(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    try:
        logger.info(f"Processing file: {file.filename}")

        token_payload = verify_token(token)
        user_role = get_user_role(token_payload)

        if not has_permission(user_role, "user"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        alter_table_add_columns()

        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()

        processors = {"pdf": process_pdf, "docx": process_docx, "txt": process_txt, "csv": process_csv, "xlsx": process_xlsx}
        if file_extension not in processors:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_processed = processors[file_extension](file_content)
        text_chunks = split_text(file_processed)
        embeddings_list = create_embeddings(text_chunks, embeddings)

        document_id, equipment_id = random.randint(1, 10000), random.randint(1, 10000)
        metadata = {"source": "blob", "blobType": file_extension}

        insert_embeddings(text_chunks, embeddings_list, document_id, equipment_id, metadata)
        return {"message": "Embeddings created and stored successfully!"}

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=1213, reload=True, debug=True)


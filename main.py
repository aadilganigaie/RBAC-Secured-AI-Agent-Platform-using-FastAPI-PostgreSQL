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

# Update CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:44360",
        "https://localhost:44360",
        "https://192.168.0.194:44360",
        "https://llm.platform.myw.ai",
        "https://rossana.platform.myw.ai"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS", "HEAD"],
    allow_headers=[
        "Authorization", 
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With"
    ],
    expose_headers=["*"],
    max_age=3600
)

# Update error handling middleware
@app.middleware("http")
async def add_error_handling(request: Request, call_next):
    try:
        # Log incoming request details
        logger.debug(f"Incoming request: {request.method} {request.url}")
        logger.debug(f"Headers: {request.headers}")
        
        response = await call_next(request)
        
        # Log response status
        logger.debug(f"Response status: {response.status_code}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://sestrilevante.platform.myw.ai:11434')

CONNECTION_STRING = "postgresql+psycopg2://mywai:Bth-12345@localhost:5432/experiment"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Update the Auth0 configuration section
AUTH0_CONFIG = {
    "Domain": "mywai-dev.eu.auth0.com",
    "Audience": "oVyE1XrDlYIClP8IDA3vrhM7Dpui7dIq",  # From token aud claim
    "Issuer": "https://mywai-dev.eu.auth0.com/",      # From token iss claim
}

ALGORITHMS = ["RS256"]

# Fetch Auth0 public key
def get_auth0_public_key():
    url = f"https://{AUTH0_CONFIG['Domain']}/.well-known/jwks.json"
    response = requests.get(url)
    return response.json()

def process_pdf(file):
    with open("temp_pdf.pdf", "wb") as f:
        f.write(file)
    loader = PyPDFLoader("temp_pdf.pdf")
    pdf_text = loader.load()
    return pdf_text

def process_docx(file):
    with open("temp_docx.docx", "wb") as f:
        f.write(file)
    loader = UnstructuredWordDocumentLoader("temp_docx.docx")
    docx_text = loader.load()
    return docx_text

def process_txt(file):
    with open("temp_txt.txt", "wb") as f:
        f.write(file)
    loader = TextLoader("temp_txt.txt")
    txt_text = loader.load()
    return txt_text

def process_csv(file):
    with open("temp_csv.csv", "wb") as f:
        f.write(file)
    loader = CSVLoader("temp_csv.csv")
    csv_text = loader.load()
    return csv_text

def process_xlsx(file):
    with open("temp_xlsx.xlsx", "wb") as f:
        f.write(file)
    loader = UnstructuredExcelLoader("temp_xlsx.xlsx")
    xlsx_text = loader.load()
    return xlsx_text

def split_text(docs, chunk_size=3000, overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))
    return chunks

def create_embeddings(text_chunks, embeddings):
    return [embeddings.embed_query(chunk) for chunk in text_chunks]

Base = declarative_base()

class LangchainPgEmbedding(Base):
    __tablename__ = 'langchain_pg_embedding'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pageContent = Column(String, name='pageContent')
    file_metadata = Column(JSONB, name='metadata')
    embedding = Column(Vector, name='embedding')
    equipment_id = Column(Integer, name='equipment_id')
    document_id = Column(Integer, name='document_id')

# Connect to PostgreSQL
engine = create_engine(CONNECTION_STRING)
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

def alter_table_add_columns():
    with engine.connect() as connection:
        metadata = MetaData()
        text_embeddings_table = Table('langchain_pg_embedding', metadata, autoload_with=engine)
        if not any(col.name == 'document_id' for col in text_embeddings_table.columns):
            connection.execute(text('ALTER TABLE langchain_pg_embedding ADD COLUMN document_id INTEGER'))
        if not any(col.name == 'equipment_id' for col in text_embeddings_table.columns):
            connection.execute(text('ALTER TABLE langchain_pg_embedding ADD COLUMN equipment_id INTEGER'))
        if not any(col.name == 'metadata' for col in text_embeddings_table.columns):
            connection.execute(text('ALTER TABLE langchain_pg_embedding ADD COLUMN metadata JSONB'))

def build_metadata(pdf_docs, file_name):
    first_doc = pdf_docs[0].metadata
    return {
        "loc": {"pageNumber": 1},
        "pdf": {
            "info": first_doc.get('info', {}),
            "version": first_doc.get('version', ''),
            "metadata": first_doc.get('metadata', {}),
            "totalPages": len(pdf_docs)
        },
        "source": "blob",
        "blobType": file_name.split('.')[-1].lower()
    }

def insert_embeddings(text_chunks, embeddings_list, document_id, equipment_id, metadata):
    for content, embedding in zip(text_chunks, embeddings_list):
        sanitized_content = content.replace('\x00', '')
        chunk_id = uuid.uuid4()
        exists = session.query(LangchainPgEmbedding).filter_by(pageContent=sanitized_content).first()
        if exists:
            continue
        new_entry = LangchainPgEmbedding(
            id=chunk_id,
            pageContent=sanitized_content,
            file_metadata=metadata,
            embedding=np.array(embedding).flatten().tolist(),
            equipment_id=equipment_id,
            document_id=document_id
        )
        session.add(new_entry)
    session.commit()

class GenerateEmbeddingsResponse(BaseModel):
    message: str

# Update the verify_token function

def verify_token(token: str):
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token.split(' ')[1]
        
        # Get token header and claims for debugging
        header = jwt.get_unverified_header(token)
        unverified_claims = jwt.get_unverified_claims(token)
        print("Token header:", json.dumps(header, indent=2))
        print("Token claims:", json.dumps(unverified_claims, indent=2))
        
        # Fetch Auth0 public keys
        jwks = get_auth0_public_key()
        
        # Find the matching key
        rsa_key = {}
        for key in jwks['keys']:
            if key['kid'] == header['kid']:
                rsa_key = {
                    'kty': key['kty'],
                    'kid': key['kid'],
                    'n': key['n'],
                    'e': key['e']
                }
                break
        
        if not rsa_key:
            raise JWTError('No matching key found')
        
        # Verify token with MyWAI specific settings
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            options={
                'verify_aud': True,      # Enable audience verification
                'verify_iss': True,      # Enable issuer verification
                'verify_exp': True,      # Check expiration
                'verify_at_hash': False  # Skip at_hash verification
            },
            audience=AUTH0_CONFIG["Audience"],
            issuer=AUTH0_CONFIG["Issuer"]
        )
        
        # Check MyWAI tenant access
        tenants = payload.get('https://mywai.com/mywai_tenants', [])
        if not tenants:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No MyWAI tenant access"
            )
        
        # Add tenant info to response
        payload['active_tenants'] = [t['name'] for t in tenants]
        
        return payload
        
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except JWTError as e:
        print(f"JWT Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        print(f"Auth Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )

@app.options("/generate_embeddings")
async def options_generate_embeddings():
    return JSONResponse(
        status_code=200,
        content={"message": "OK"}
    )

from fastapi import Depends, HTTPException, status

def get_user_role(token_payload):
    """Extract role from MyWAI token payload"""
    # Get roles from MyWAI custom claim
    mywai_roles = token_payload.get('https://mywai.com/roles', [])
    
    # Check if user has any tenants (indicating they are at least a user)
    mywai_tenants = token_payload.get('https://mywai.com/mywai_tenants', [])
    
    if mywai_roles:  # If roles are explicitly defined
        if 'admin' in mywai_roles:
            return 'admin'
        elif any(role for role in mywai_roles):
            return 'user'
    elif mywai_tenants:  # If user has tenant access, treat as user
        return 'user'
        
    return 'guest'

def has_permission(role, required_role):
    """Check if role meets required permission level"""
    role_hierarchy = {
        'admin': 3,
        'user': 2,
        'guest': 1
    }
    return role_hierarchy.get(role, 0) >= role_hierarchy.get(required_role, 0)

@app.post("/generate_embeddings", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Log request start
        logger.info(f"Processing file: {file.filename}")

        # Verify token and get user info
        token_payload = verify_token(token)
        logger.info(f"Authenticated user: {token_payload.get('name')}")
        logger.info(f"Active tenants: {token_payload.get('active_tenants')}")

        # Check user role - consider user with any tenant access as valid user
        user_role = get_user_role(token_payload)
        logger.debug(f"User role determined as: {user_role}")
        
        if not has_permission(user_role, 'user'):
            logger.warning(f"Permission denied for user {token_payload.get('name')} with role {user_role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions - User role required"
            )

        if not file:
            raise HTTPException(status_code=400, detail="File is required")

        alter_table_add_columns()

        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()

        try:
            if file_extension == 'pdf':
                file_processed = process_pdf(file_content)
            elif file_extension == 'docx':
                file_processed = process_docx(file_content)
            elif file_extension == 'txt':
                file_processed = process_txt(file_content)
            elif file_extension == 'csv':
                file_processed = process_csv(file_content)
            elif file_extension == 'xlsx':
                file_processed = process_xlsx(file_content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            text_chunks = split_text(file_processed)
            embeddings_list = create_embeddings(text_chunks, embeddings)

            document_id = random.randint(1, 10000)
            equipment_id = random.randint(1, 10000)

            metadata = build_metadata(file_processed, file.filename)

            insert_embeddings(text_chunks, embeddings_list, document_id, equipment_id, metadata)
            return GenerateEmbeddingsResponse(message="Embeddings created and stored successfully!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=1213, reload=True, debug=True)

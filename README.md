# RBAC-Secured AI Agent Platform using FastAPI & PostgreSQL

## Overview
This repository contains a **Role-Based Access Control (RBAC) implementation** for securing AI agent interactions. Built with **FastAPI, PostgreSQL, and JWT authentication**, this platform ensures fine-grained access control, secure document processing, and AI-driven embeddings using open-source LLMs.

## Features
- **FastAPI Backend** for handling AI agent requests
- **RBAC Enforcement** with Admin, User, and Guest roles
- **OAuth 2.0 & JWT Authentication** for secure access
- **Document Processing** (PDF, DOCX, TXT, CSV, XLSX) for AI embeddings
- **Embeddings Storage** using PostgreSQL with pgvector extension
- **Logging & Security Auditing** for tracking API usage
- **CORS Support** for secure API calls

## Tech Stack
- **FastAPI** - Web framework
- **PostgreSQL** - Database with pgvector support
- **LangChain & Ollama** - AI Embeddings
- **OAuth2 & JWT** - Authentication
- **SQLAlchemy** - ORM for database interactions
- **Uvicorn** - ASGI server for FastAPI

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- PostgreSQL with `pgvector`
- Virtual environment (`venv`)

### Clone Repository
```bash
git clone https://github.com/yourusername/rbac-secured-ai-platform.git
cd rbac-secured-ai-platform
```

### Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # For Windows use: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration
### Environment Variables
Create a `.env` file and configure:
```env
DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/dbname
AUTH0_DOMAIN=your-auth0-domain
AUTH0_AUDIENCE=your-audience
JWT_ALGORITHM=RS256
```

## Running the Application
Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints
| Method | Endpoint | Description |
|--------|---------|-------------|
| POST   | `/generate_embeddings` | Upload document for AI processing |
| OPTIONS | `/generate_embeddings` | Preflight request for CORS |

## Role-Based Access Control (RBAC)
| Role  | Permissions |
|-------|------------|
| Admin | Full access |
| User  | Can process documents |
| Guest | Limited access |

## Logging & Security
- All authentication attempts are logged
- Unauthorized access attempts trigger security alerts

## Deployment
For production deployment, consider using **Docker**:
```bash
docker build -t rbac-ai-agent .
docker run -p 8000:8000 rbac-ai-agent
```

## License
This project is licensed under the MIT License.

## Contributors
- [Your Name](https://github.com/yourusername)

## Contact
For any issues or feature requests, create a GitHub issue.

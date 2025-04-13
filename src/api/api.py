from fastapi import FastAPI
from pydantic import BaseModel
import re

# Load encapsulated API only
from api.chatgpt_api import ChatGPTAPI

# FastAPI application initialization
app = FastAPI(title="RAG-LLM-SST API")

# Loading the main component
chat = ChatGPTAPI()

# POST request data model
class Query(BaseModel):
    query: str

# Welcome GET route
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API RAG-LLM-SST 🧠. Utilisez /query pour poser une question."
    }

# Main POST route for querying the model
@app.post("/query")
def query_llm(input: Query):
    user_query = input.query.strip().lower()

    # Social or meta query management
    if re.match(r"^(bonjour|salut|coucou|qui (es|êtes)-tu|merci|au revoir|hello)", user_query):
        return {
            "response": (
                "Bonjour 👋 Je suis un assistant spécialisé en SST (Sauveteur Secouriste du Travail). "
                "Posez-moi une question sur les gestes de premiers secours, la prévention ou la conduite à tenir en cas d'accident."
            )
        }

    # Direct response generation (the retriever is called in ChatGPTAPI)
    response = chat.get_response(user_query)
    return {"response": response}






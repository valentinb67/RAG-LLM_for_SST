import os
import sys
import openai
from dotenv import load_dotenv

# Add src/ to find model_management
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from model_management.hybrid_retrieval import HybridRetriever

# Determine the absolute path to the project root
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
index_path = os.path.join(base_dir, "models", "index", "index_files", "faiss_index")
metadata_path = os.path.join(base_dir, "models", "index", "index_files", "metadata_passages.json")

# Load .env
env_path = os.path.join(base_dir, "config", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    raise FileNotFoundError(f"The .env file cannot be found at the location {env_path}")

# LOad API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key missing. Check .env file.")

openai.api_key = OPENAI_API_KEY

# Initialize retriever with absolute paths
retriever = HybridRetriever(index_path=index_path, embeddings_path=metadata_path)

# Interface class for interacting with the OpenAI API
class ChatGPTAPI:
    def __init__(self, model=OPENAI_MODEL):
        self.model = model

    def build_prompt(self, question, context):
        return (
            "Tu es un assistant expert en SST (Sauveteur Secouriste du Travail). "
            "Réponds de manière précise et factuelle en t'appuyant sur les informations suivantes, "
            "et cite tes sources (document, page, section, sous-section) si possible.\n\n"
            f" Contexte :\n{context}\n\n"
            f" Question :\n{question}\n\n"
            " **Réponse détaillée :**"
        )

    def get_response(self, question):
        try:
            context = retriever.retrieve_hybrid(query=question)
            if not context:
                context = "Aucune information spécifique trouvée dans la base de données."

            prompt = self.build_prompt(question, context)

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en SST."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            return response["choices"][0]["message"]["content"].strip()

        except openai.error.AuthenticationError:
            return " Authentication error: invalid or missing API key."
        except openai.error.OpenAIError as e:
            return f" OpenAI error: {str(e)}"
        except Exception as e:
            return f" Unknown error : {str(e)}"







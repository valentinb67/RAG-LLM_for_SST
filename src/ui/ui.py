import gradio as gr
import requests

# URL of FastAPI
API_URL = "http://127.0.0.1:8000/ask"

def chat_with_api(user_input, history=[]):
    """
    Fonction pour envoyer une question à l'API et récupérer la réponse.
    """
    response = requests.post(API_URL, json={"question": user_input})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Erreur API : Vérifiez que le serveur est bien lancé."

# Gradio interface
with gr.Blocks(theme="default") as app:
    with gr.Row():
        gr.Image("https://cdn-icons-png.flaticon.com/512/2206/2206368.png", width=60, height=60)  # incon of the Chatbot 
        gr.HTML("<h1 style='margin-left: 10px;'>💬 Chatbot SST - RAG ChatGPT</h1>")

    gr.Markdown("### Posez une question et obtenez une réponse basée sur les documents indexés 🏗️")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_api,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Tapez votre question ici...", lines=2),
        submit_btn="Envoyer 🚀",
        clear_btn="Effacer 🗑️"
    )

# Launch
if __name__ == "__main__":
    app.launch()




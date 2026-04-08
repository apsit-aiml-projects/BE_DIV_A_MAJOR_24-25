from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# Allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request format
class ChatRequest(BaseModel):
    message: str

# Simple rule-based chatbot responses
responses = {
    "hello": ["Hi there!", "Hello!", "Hey! How can I help you?"],
    "how are you": ["I'm just a bot, but I'm doing fine!", "I'm good! What about you?"],
    "bye": ["Goodbye!", "See you later!", "Take care!"],
    "default": ["I'm not sure how to respond to that.", "Can you rephrase your question?"]
}

def chatbot_response(user_message):
    """Find a response based on the user's message."""
    user_message = user_message.lower()
    for key in responses:
        if key in user_message:
            return random.choice(responses[key])
    return random.choice(responses["default"])

@app.post("/chat")
async def chatbot(request: ChatRequest):
    bot_reply = chatbot_response(request.message)
    return {"reply": bot_reply}

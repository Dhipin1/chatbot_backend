import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(title="Chatbot Backend")


class ChatRequest(BaseModel):
    message: str
    diagnosis_context: dict | None = None


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def home():
    return {"message": "Chatbot backend is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    system_prompt = (
        "You are a helpful chatbot inside a plant disease diagnosis app. "
        "Answer clearly and safely. "
        "If diagnosis context is provided, use it in your answer. "
        "If medical/agricultural certainty is not possible, say it is a possible suggestion."
    )

    context_text = ""
    if req.diagnosis_context:
        context_text = f"\nDiagnosis Context:\n{req.diagnosis_context}\n"

    full_prompt = f"""
{system_prompt}

{context_text}

User message:
{req.message}
"""

    response = model.generate_content(full_prompt)
    return ChatResponse(reply=response.text)
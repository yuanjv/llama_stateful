from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uuid
from llama_agent import agent  # Import our backend

app = FastAPI()

# --- Request Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # Optional for new sessions

class StartSessionResponse(BaseModel):
    session_id: str

class ChatResponse(BaseModel):
    response: str

# --- Session Management Endpoints ---
@app.post("/start_session", response_model=StartSessionResponse)
def start_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    try:
        agent.start_session(session_id)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(500, f"Session creation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_session_id: str = Header(None)
):
    """Process user message using session KV cache"""
    session_id = req.session_id or x_session_id
    if not session_id:
        raise HTTPException(400, "Missing session ID")
    
    try:
        response = agent.chat(session_id, req.message)
        return {"response": response}
    except ValueError:
        raise HTTPException(404, "Invalid session ID")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.delete("/sessions/{session_id}")
def end_session(session_id: str):
    """Terminate session and free resources"""
    if agent.end_session(session_id):
        return {"status": "session ended"}
    raise HTTPException(404, "Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
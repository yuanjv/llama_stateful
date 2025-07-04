from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from llama_agent import session_manager  # Import our backend

app = FastAPI()

# --- Request Models ---
class ChatRequest(BaseModel):
    message: str

class SessionResponse(BaseModel):
    session_id: str

class ChatResponse(BaseModel):
    response: str

# --- Session Endpoints ---
@app.post("/sessions", response_model=SessionResponse)
def create_session():
    """Create new chat session with KV cache"""
    try:
        session_id = session_manager.create_session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(500, f"Session creation failed: {str(e)}")

@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat_message(session_id: str, req: ChatRequest):
    """Process user message using session KV cache"""
    try:
        response = session_manager.process_message(session_id, req.message)
        return {"response": response}
    except ValueError:
        raise HTTPException(404, "Invalid session ID")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Terminate session and free resources"""
    if session_manager.end_session(session_id):
        return {"status": "session ended"}
    raise HTTPException(404, "Session not found")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "active_sessions": len(session_manager.sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
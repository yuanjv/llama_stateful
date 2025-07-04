import threading
from llama_cpp import Llama
from llama_cpp_agent.chat_history import BasicChatMessageStore

class LlamaAgent:
    def __init__(self, model_path, context_size=2048):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=-1,
            use_mlock=True,
            n_threads=4,
            offload_kqv=True
        )
        self.lock = threading.Lock()
        self.message_store = BasicChatMessageStore()
        self.sessions = {}  # session_id: state object
        
    def start_session(self, session_id):
        """Initialize new session with empty KV cache"""
        with self.lock:
            if session_id in self.sessions:
                raise ValueError("Session already exists")
            
            # Initialize state and history
            self.sessions[session_id] = self.llm.create_state()
            self.message_store.add_chat_history(session_id)
            
            # Add system prompt
            self.message_store.add_message(
                session_id, 
                "system", 
                "You are a helpful assistant. Keep responses concise and clear."
            )
        return session_id
    
    def chat(self, session_id, message):
        """Process message using session's KV cache"""
        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")
        
        # Add user message to history
        self.message_store.add_message(session_id, "user", message)
        
        with self.lock:
            state = self.sessions[session_id]
            
            # Generate response using cached state
            output = self.llm.create_completion(
                prompt=message,
                state=state,
                max_tokens=200,
                temperature=0.7,
                stop=["\n", "###"],
                stream=False
            )
            
            # Update session state
            self.sessions[session_id] = state
            
            reply = output['choices'][0]['text']
        
        # Save assistant response
        self.message_store.add_message(session_id, "assistant", reply)
        return reply
    
    def end_session(self, session_id):
        """Clean up session resources"""
        if session_id not in self.sessions:
            return False
            
        with self.lock:
            # Free resources (state cleanup handled by GC)
            del self.sessions[session_id]
            self.message_store.delete_chat_history(session_id)
        return True

# Global agent instance (singleton pattern)
MODEL_PATH = "./models/your_model.gguf"  # UPDATE THIS
agent = LlamaAgent(MODEL_PATH)
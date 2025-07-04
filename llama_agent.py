from llama_cpp import Llama
import uuid

class SessionManager:
    def __init__(self, model_path, context_size=2048, n_parallel=5):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=-1,
            use_mlock=True,
            n_threads=8,  # More threads for parallel processing
            n_parallel=n_parallel,  # Key parameter for concurrency
            offload_kqv=True,
            verbose=False
        )
        self.sessions = {}
        self.system_prompt = "You are a helpful assistant. Keep responses concise and clear."
    
    def create_session(self):
        """Initialize new session with empty KV cache"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'state': self.llm.create_state(),
            'history': [
                {"role": "system", "content": self.system_prompt}
            ]
        }
        
        # Initialize state with system prompt
        self._eval_prompt(session_id, self.system_prompt)
        return session_id
    
    def _eval_prompt(self, session_id, prompt):
        """Process prompt through model without generation"""
        session = self.sessions[session_id]
        self.llm.eval(
            tokens=self.llm.tokenize(prompt.encode("utf-8")),
            state=session['state']
        )
    
    def process_message(self, session_id, message):
        """Handle message using session's KV cache"""
        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")
        
        session = self.sessions[session_id]
        
        # Format user message and update history
        user_prompt = f"User: {message}\nAssistant:"
        session['history'].append({"role": "user", "content": message})
        
        # Process user message through model
        self._eval_prompt(session_id, user_prompt)
        
        # Generate response
        output = self.llm.create_completion(
            prompt="",  # Use existing state
            state=session['state'],
            max_tokens=200,
            temperature=0.7,
            stop=["\n", "###"],
            stream=False
        )
        
        # Extract and store response
        reply = output['choices'][0]['text'].strip()
        session['history'].append({"role": "assistant", "content": reply})
        
        return reply
    
    def end_session(self, session_id):
        """Clean up session resources"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

# Global session manager (thread-safe through llama.cpp parallel processing)
MODEL_PATH = "./models/your_model.gguf"  # UPDATE THIS
session_manager = SessionManager(MODEL_PATH, n_parallel=5)
#!/usr/bin/env python3
"""
Flask web application for the Agentic Data Assistant.
This creates a web interface that connects to the existing ChatbotPipeline.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import sys
from datetime import datetime
import traceback
import uuid
from typing import Dict, List, Any

# Import the existing chatbot pipeline
from app import ChatbotPipeline

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-in-production')

# Initialize the chatbot pipeline (this may take a moment as it loads data)
print("🚀 Initializing Agentic Data Assistant...")
chatbot = None

# In-memory session storage for conversation history
# In production, this should be replaced with a database or Redis
session_histories: Dict[str, List[Dict[str, Any]]] = {}

def initialize_chatbot():
    """Initialize the chatbot pipeline with error handling."""
    global chatbot
    try:
        chatbot = ChatbotPipeline()
        print("✅ Chatbot initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("Full error traceback:")
        traceback.print_exc()
        return False

def get_or_create_session_id() -> str:
    """Get existing session ID or create a new one."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    """Get conversation history for a session."""
    if session_id not in session_histories:
        session_histories[session_id] = []
    return session_histories[session_id]

def add_to_conversation_history(session_id: str, user_message: str, bot_response: str, token_usage: Dict = None):
    """Add a message exchange to conversation history."""
    if session_id not in session_histories:
        session_histories[session_id] = []
    
    conversation_entry = {
        'user_message': user_message,
        'bot_response': bot_response,
        'timestamp': datetime.now().isoformat(),
        'token_usage': token_usage or {}
    }
    
    session_histories[session_id].append(conversation_entry)
    
    # Keep only the last 10 conversation exchanges to manage memory
    if len(session_histories[session_id]) > 10:
        session_histories[session_id] = session_histories[session_id][-10:]

def format_conversation_history_for_agents(history: List[Dict[str, Any]], enable_context_history: bool = True) -> str:
    """Format conversation history for agent consumption."""
    if not history or not enable_context_history:
        return "This is the start of a new conversation."
    
    formatted_history = ["Previous conversation context:"]
    for i, entry in enumerate(history[-5:], 1):  # Only include last 5 exchanges
        formatted_history.append(f"Exchange {i}:")
        formatted_history.append(f"User: {entry['user_message']}")
        formatted_history.append(f"Assistant: {entry['bot_response'][:200]}...")  # Truncate long responses
        formatted_history.append("")
    
    return "\n".join(formatted_history)

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the frontend."""
    global chatbot
    
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message'].strip()
        # Get toggle states with defaults
        enable_grounding = data.get('enable_grounding', True)
        enable_context_history = data.get('enable_context_history', True)
        # Get session ID from request or create new one
        session_id = data.get('session_id') or get_or_create_session_id()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        # Initialize chatbot if not already done
        if chatbot is None:
            if not initialize_chatbot():
                return jsonify({
                    'success': False,
                    'error': 'Chatbot is not available. Please check server logs.'
                }), 500
        
        print(f"📤 Processing query: '{user_message}' (session: {session_id[:8]}, grounding: {'ON' if enable_grounding else 'OFF'}, context: {'ON' if enable_context_history else 'OFF'})")
        
        # Get conversation history for this session
        conversation_history = get_conversation_history(session_id)
        formatted_history = format_conversation_history_for_agents(conversation_history, enable_context_history)
        
        # Process the message through the chatbot pipeline with conversation history
        chatbot_response = chatbot.run(
            user_message, 
            enable_grounding=enable_grounding,
            conversation_history=formatted_history,
            session_id=session_id
        )

        if len(chatbot_response) == 2:
            result, response = chatbot_response
        else:
            result = chatbot_response
        
        # Handle the new dictionary return format
        if isinstance(result, dict):
            response_text = result.get("answer", "No answer provided")
            token_usage = result.get("token_usage", {})
            agent_breakdown = result.get("agent_token_breakdown", {})
        else:
            # Fallback for backward compatibility
            response_text = result
            token_usage = {}
            agent_breakdown = {}
        
        # Add this exchange to conversation history
        add_to_conversation_history(session_id, user_message, response_text, token_usage)
        
        print(f"📥 Response generated successfully")
        print(f"🔢 Token usage: {token_usage}")
        
        return jsonify({
            'success': True,
            'response': response_text,
            'session_id': session_id,
            'token_usage': token_usage,
            'agent_token_breakdown': agent_breakdown,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error processing chat message: {e}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'An error occurred while processing your message: {str(e)}'
        }), 500

@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new conversation session."""
    new_session_id = str(uuid.uuid4())
    session['session_id'] = new_session_id
    return jsonify({
        'success': True,
        'session_id': new_session_id,
        'message': 'New conversation started'
    })

@app.route('/session_history', methods=['GET'])
def get_session_history():
    """Get conversation history for current session."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({
            'success': False,
            'error': 'No active session'
        }), 400
    
    history = get_conversation_history(session_id)
    return jsonify({
        'success': True,
        'session_id': session_id,
        'history': history
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'chatbot_ready': chatbot is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Query the model"""
    global response
    if request.method == 'POST':
        data = request.json

        if data and 'user_id' in data:
            session_id = data['user_id'] or get_or_create_session_id()
        if data and 'message' in data:
            user_message = data['message']
        else: 
            return jsonify({'success': False,'error': 'Empty message'}), 400
        
        if chatbot is None:
            if not initialize_chatbot():
                return jsonify({
                    'success': False,
                    'error': 'Chatbot is not available. Please check server logs.'
                }), 500
        
        # Get conversation history for this session
        conversation_history = get_conversation_history(session_id)
        formatted_history = format_conversation_history_for_agents(conversation_history)
        
        # Process the message through the chatbot pipeline with conversation history
        chatbot_response = chatbot.run(
            user_message, 
            enable_grounding=False,
            conversation_history=formatted_history,
            session_id=session_id
        )
        if len(chatbot_response) == 2:
            result_body, response_body = chatbot_response
            # Handle the new dictionary return format
            if isinstance(result_body, dict):
                response_text = result_body.get("answer", "No answer provided")
                token_usage = result_body.get("token_usage", {})
            else:
                # Fallback for backward compatibility
                response_text = result_body
                token_usage = {}
        
            # Add this exchange to conversation history
            add_to_conversation_history(session_id, user_message, response_text, token_usage)
        
            print(f"📥 Response generated successfully")
            print(f"🔢 Token usage: {token_usage}")
            response = response_body
            return response_body
        else:
            return chatbot_response
    elif request.method == 'GET':
        return response
    else:
        return jsonify({"error": "Wrong method invoked."})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize chatbot on startup
    print("🔧 Setting up Agentic Data Assistant...")
    
    # Check if we're in the right directory
    if False:  # logo check removed
    
    if not os.path.exists('static'):
        os.makedirs('static')
        print("📁 Created static directory")
    
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates directory")
    
    # Initialize the chatbot
    initialize_chatbot()
    
    # Start the Flask development server
    print("\n🌐 Starting Agentic Data Assistant...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    ) 
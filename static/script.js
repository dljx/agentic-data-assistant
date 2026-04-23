// Global variables
let isProcessing = false;
let currentSessionId = null;

// DOM elements
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const loadingIndicator = document.getElementById('loadingIndicator');

// Session management functions
function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

function getCurrentSessionId() {
    if (!currentSessionId) {
        currentSessionId = sessionStorage.getItem('session_id') || generateSessionId();
        sessionStorage.setItem('session_id', currentSessionId);
    }
    return currentSessionId;
}

function startNewSession() {
    currentSessionId = generateSessionId();
    sessionStorage.setItem('session_id', currentSessionId);
    
    // Clear chat messages
    clearChat();
    
    // Add welcome message
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'welcome-message';
    welcomeDiv.innerHTML = '<p>New conversation started! Ask me anything about companies, investors, and market data.</p>';
    chatMessages.appendChild(welcomeDiv);
    
    console.log('Started new session:', currentSessionId);
}

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    // Initialize session
    getCurrentSessionId();
    console.log('Current session:', currentSessionId);
    
    // Focus on input when page loads
    messageInput.focus();
    
    // Add event listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // New conversation button
    const newConversationBtn = document.getElementById('newConversationBtn');
    if (newConversationBtn) {
        newConversationBtn.addEventListener('click', startNewSession);
    }
    
    // Auto-resize input and maintain focus
    messageInput.addEventListener('input', function() {
        updateSendButtonState();
    });
    
    // Initial button state
    updateSendButtonState();

    // Set up grounding toggle event listener
    const groundingToggle = document.getElementById('groundingToggle');
    const groundingStatus = document.getElementById('groundingStatus');
    
    if (groundingToggle && groundingStatus) {
        // Function to update status text
        function updateGroundingStatus() {
            if (groundingToggle.checked) {
                groundingStatus.textContent = '🌐 Web search enabled';
                groundingStatus.classList.remove('disabled');
            } else {
                groundingStatus.textContent = '🔒 Web search disabled';
                groundingStatus.classList.add('disabled');
            }
        }
        
        // Update status on toggle change
        groundingToggle.addEventListener('change', updateGroundingStatus);
        
        // Set initial status
        updateGroundingStatus();
    }
    
    // Set up context history toggle event listener
    const contextHistoryToggle = document.getElementById('contextHistoryToggle');
    const contextHistoryStatus = document.getElementById('contextHistoryStatus');
    
    if (contextHistoryToggle && contextHistoryStatus) {
        // Function to update status text
        function updateContextHistoryStatus() {
            if (contextHistoryToggle.checked) {
                contextHistoryStatus.textContent = '💭 Conversation memory enabled';
                contextHistoryStatus.classList.remove('disabled');
            } else {
                contextHistoryStatus.textContent = '🔄 Fresh start each time';
                contextHistoryStatus.classList.add('disabled');
            }
        }
        
        // Update status on toggle change
        contextHistoryToggle.addEventListener('change', updateContextHistoryStatus);
        
        // Set initial status
        updateContextHistoryStatus();
    }
    
    // Add keyboard shortcut for new conversation
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + R for new conversation
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            startNewSession();
        }
    });
});

// Update send button state based on input
function updateSendButtonState() {
    const hasText = messageInput.value.trim().length > 0;
    sendButton.disabled = !hasText || isProcessing;
}

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) {
        return;
    }
    
    // Prevent double-sending
    isProcessing = true;
    updateSendButtonState();
    
    // Clear input
    messageInput.value = '';
    
    // Remove welcome message if it exists
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show loading indicator
    showLoading();
    
    try {
        // Get toggle states
        const groundingToggle = document.getElementById('groundingToggle');
        const contextHistoryToggle = document.getElementById('contextHistoryToggle');
        const enableGrounding = groundingToggle ? groundingToggle.checked : true;
        const enableContextHistory = contextHistoryToggle ? contextHistoryToggle.checked : true;
        
        // Send request to backend with toggle states and session ID
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                enable_grounding: enableGrounding,
                enable_context_history: enableContextHistory,
                session_id: getCurrentSessionId()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide loading indicator
        hideLoading();
        
        // Update session ID if provided by server
        if (data.session_id) {
            currentSessionId = data.session_id;
            sessionStorage.setItem('session_id', currentSessionId);
        }
        
        // Add bot response to chat with enhanced formatting
        if (data.success) {
            addMessage(data.response, 'bot', data.token_usage, data.agent_token_breakdown, message);
        } else {
            addMessage(`Sorry, I encountered an error: ${data.error}`, 'bot');
        }
        
    } catch (error) {
        console.error('Error:', error);
        hideLoading();
        addMessage('Sorry, I\'m having trouble connecting right now. Please try again.', 'bot');
    }
    
    // Reset processing state
    isProcessing = false;
    updateSendButtonState();
    messageInput.focus();
}

// Enhanced message formatting function
function formatText(text) {
    if (!text) return '';
    
    // Escape HTML first to prevent XSS
    text = escapeHtml(text);
    
    // Convert markdown-style formatting
    text = text
        // Headers
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        
        // Bold and italic
        .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        
        // Code blocks
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        
        // Lists
        .replace(/^\* (.+$)/gm, '<li>$1</li>')
        .replace(/^- (.+$)/gm, '<li>$1</li>')
        .replace(/^\d+\. (.+$)/gm, '<li>$1</li>')
        
        // Line breaks and paragraphs
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    // Wrap in paragraph tags if not already wrapped
    if (!text.includes('<p>') && !text.includes('<h1>') && !text.includes('<h2>') && !text.includes('<h3>')) {
        text = '<p>' + text + '</p>';
    }
    
    // Wrap lists in ul tags
    if (text.includes('<li>')) {
        text = text.replace(/(<li>.*?<\/li>)/gs, function(match) {
            return '<ul>' + match + '</ul>';
        });
    }
    
    return text;
}

// Add message to chat with enhanced formatting
function addMessage(text, sender, token_usage, agent_token_breakdown, lastMessage) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? 'You' : 'AI';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    if (sender === 'bot') {
        // Apply enhanced formatting for bot messages
        content.innerHTML = formatText(text);
        
        // Add token usage information if available (simplified version)
        if (token_usage && (token_usage.total_tokens > 0)) {
            const tokenDiv = document.createElement('div');
            tokenDiv.className = 'token-usage';
            
            // Calculate adjusted completion tokens to include overhead
            const adjustedCompletion = token_usage.total_tokens - token_usage.prompt_tokens;

            // Calculate cost
            input_cost = parseFloat(((token_usage.prompt_tokens/1000) * 0.0003).toFixed(6))
            output_cost = parseFloat(((adjustedCompletion/1000) * 0.0025).toFixed(6))
            total_cost = parseFloat((input_cost + output_cost).toFixed(6))
            
            tokenDiv.innerHTML = `
                <div class="token-summary">
                    <span class="token-icon">🔢</span>
                    <span class="token-text">
                        <strong>${token_usage.total_tokens}</strong> tokens used
                    </span>
                </div>
                <div class="token-breakdown">
                    <details>
                        <summary>Token Breakdown</summary>
                        <div class="breakdown-content">
                            <div class="token-detail">
                                <span class="token-type">Input:</span>
                                <span class="token-count">${token_usage.prompt_tokens}</span>
                            </div>
                            <div class="token-detail">
                                <span class="token-type">Output:</span>
                                <span class="token-count">${adjustedCompletion}</span>
                            </div>
                            <div class="token-detail">
                                <span class="token-type">Input Cost: </span>
                                <span class="token-count">$${input_cost}</span>
                            </div>
                            <div class="token-detail">
                                <span class="token-type">Output Cost: </span>
                                <span class="token-count">$${output_cost}</span>
                            </div>
                            <div class="token-detail">
                                <span class="token-type">Total Cost: </span>
                                <span class="token-count">$${total_cost}</span>
                            </div>
                        </div>
                    </details>
                </div>
            `;
            
            content.appendChild(tokenDiv);
        }
    } else {
        // Keep user messages as plain text
        content.textContent = text;
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom with smooth animation
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
}

// Show loading indicator
function showLoading() {
    loadingIndicator.classList.add('show');
}

// Hide loading indicator
function hideLoading() {
    loadingIndicator.classList.remove('show');
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle typing indicator (optional enhancement)
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = '<div class="typing-dots"><div></div><div></div><div></div></div>';
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(content);
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Error handling for network issues
window.addEventListener('online', function() {
    console.log('Connection restored');
});

window.addEventListener('offline', function() {
    console.log('Connection lost');
    if (isProcessing) {
        hideLoading();
        addMessage('Connection lost. Please check your internet connection and try again.', 'bot');
        isProcessing = false;
        updateSendButtonState();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Focus input with Ctrl/Cmd + K
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        messageInput.focus();
    }
    
    // Clear chat with Ctrl/Cmd + L
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        clearChat();
    }
});

// Clear chat function
function clearChat() {
    // Remove all messages except system elements
    const messages = chatMessages.querySelectorAll('.message, .welcome-message');
    messages.forEach(message => message.remove());
}

// Add enhanced CSS for better message formatting
const style = document.createElement('style');
style.textContent = `
    /* Enhanced message content styling */
    .message-content h1, .message-content h2, .message-content h3 {
        margin: 0.5em 0;
        color: #2c3e50;
    }
    
    .message-content h1 { font-size: 1.4em; font-weight: 700; }
    .message-content h2 { font-size: 1.2em; font-weight: 600; }
    .message-content h3 { font-size: 1.1em; font-weight: 600; }
    
    .message-content p {
        margin: 0.75em 0;
        line-height: 1.6;
    }
    
    .message-content p:first-child {
        margin-top: 0;
    }
    
    .message-content p:last-child {
        margin-bottom: 0;
    }
    
    .message-content ul {
        margin: 0.5em 0;
        padding-left: 1.2em;
    }
    
    .message-content li {
        margin: 0.3em 0;
        line-height: 1.5;
    }
    
    .message-content strong {
        font-weight: 600;
        color: #2c3e50;
    }
    
    .message-content em {
        font-style: italic;
        color: #5a6c7d;
    }
    
    .message-content code {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 0.2em 0.4em;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
        font-size: 0.9em;
        color: #e83e8c;
    }
    
    .message-content pre {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1em;
        margin: 0.5em 0;
        overflow-x: auto;
    }
    
    .message-content pre code {
        background: none;
        border: none;
        padding: 0;
        color: #495057;
    }
    
    /* Typing indicator styling */
    .typing-dots {
        display: flex;
        gap: 4px;
        padding: 8px 0;
    }
    
    .typing-dots div {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #999;
        animation: typing 1.4s ease-in-out infinite both;
    }
    
    .typing-dots div:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots div:nth-child(2) { animation-delay: -0.16s; }
    .typing-dots div:nth-child(3) { animation-delay: 0s; }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Smooth scrolling */
    .chat-messages {
        scroll-behavior: smooth;
    }
    
    /* Enhanced message spacing */
    .message {
        margin-bottom: 24px;
    }
    
    /* User message styling remains clean */
    .message.user .message-content {
        color: white;
    }
    
    .message.user .message-content * {
        color: inherit;
    }
`;
document.head.appendChild(style); 
// API Configuration
const API_BASE_URL = window.API_BASE_URL || '';
const ENDPOINTS = {
    health: '/health',
    chat: '/chat',
    ingest: '/ingest',
    stats: '/stats'
};

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const articleUrl = document.getElementById('article-url');
const addArticleButton = document.getElementById('add-article-button');
const articleStatus = document.getElementById('article-status');
const loadingOverlay = document.getElementById('loading-overlay');
const toastContainer = document.getElementById('toast-container');
const charCount = document.getElementById('char-count');
const articleCount = document.getElementById('article-count');
const healthStatus = document.getElementById('health-status');
const newConversationButton = document.getElementById('new-conversation-button');

// State
let isLoading = false;
let sessionId = null;

// Generate or retrieve session ID
function getSessionId() {
    if (!sessionId) {
        // Try to get from localStorage first
        sessionId = localStorage.getItem('chat_session_id');
        
        if (!sessionId) {
            // Generate new session ID
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('chat_session_id', sessionId);
        }
    }
    return sessionId;
}

// Clear session (for new conversation)
function clearSession() {
    sessionId = null;
    localStorage.removeItem('chat_session_id');
    // Clear chat container
    chatContainer.innerHTML = '';
}

// Start new conversation
function startNewConversation() {
    if (confirm('Start a new conversation? This will clear the current chat history.')) {
        clearSession();
        addWelcomeMessage();
        showToast('New conversation started', 'success');
    }
}

// Add welcome message
function addWelcomeMessage() {
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'message bot-message';
    welcomeDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                Welcome! I can help you analyze and discuss the articles in our database. Try asking:
                <ul>
                    <li>"What are the main topics in the articles?"</li>
                    <li>"Summarize the AI-related articles"</li>
                    <li>"What's the sentiment about tech companies?"</li>
                    <li>"Compare articles about Meta and Intel"</li>
                </ul>
            </div>
        </div>
    `;
    chatContainer.appendChild(welcomeDiv);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

async function initializeApp() {
    await checkHealth();
    await updateArticleCount();
}

function setupEventListeners() {
    // Chat input event listeners
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatInput.addEventListener('input', () => {
        updateCharCount();
        toggleSendButton();
    });

    sendButton.addEventListener('click', sendMessage);

    // New conversation button
    newConversationButton.addEventListener('click', startNewConversation);

    // Article management event listeners
    addArticleButton.addEventListener('click', addArticle);
    articleUrl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addArticle();
        }
    });

    articleUrl.addEventListener('input', toggleAddButton);
}

function updateCharCount() {
    const currentLength = chatInput.value.length;
    charCount.textContent = `${currentLength}/500`;
    
    if (currentLength > 400) {
        charCount.style.color = '#f56565';
    } else if (currentLength > 300) {
        charCount.style.color = '#ed8936';
    } else {
        charCount.style.color = '#a0aec0';
    }
}

function toggleSendButton() {
    const hasText = chatInput.value.trim().length > 0;
    sendButton.disabled = !hasText || isLoading;
}

function toggleAddButton() {
    const hasValidUrl = isValidUrl(articleUrl.value.trim());
    addArticleButton.disabled = !hasValidUrl || isLoading;
}

function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}${ENDPOINTS.health}`);
        const data = await response.json();
        
        if (response.ok && data.status === 'healthy') {
            healthStatus.textContent = 'Healthy';
            healthStatus.style.color = '#48bb78';
        } else {
            throw new Error('Unhealthy');
        }
    } catch (error) {
        healthStatus.textContent = 'Offline';
        healthStatus.style.color = '#f56565';
        showToast('API server is offline', 'error');
    }
}

async function updateArticleCount() {
    try {
        const response = await fetch(`${API_BASE_URL}${ENDPOINTS.stats}`);
        
        if (response.ok) {
            const data = await response.json();
            articleCount.textContent = `${data.article_count} Articles`;
        } else {
            articleCount.textContent = 'Unknown';
        }
    } catch (error) {
        articleCount.textContent = 'Error';
        console.error('Error fetching stats:', error);
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isLoading) return;

    // Add user message to chat
    addMessageToChat(message, 'user');
    
    // Add processing indicator to chat
    const processingId = addProcessingIndicator();
    
    // Clear input
    chatInput.value = '';
    updateCharCount();
    toggleSendButton();

    // Set loading state (but no overlay)
    isLoading = true;
    toggleSendButton();
    toggleAddButton();

    try {
        const response = await fetch(`${API_BASE_URL}${ENDPOINTS.chat}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: message,
                max_articles: 5,
                session_id: getSessionId()
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        
        console.log('Chat response received:', data); // Debug log
        
        // Validate response structure
        if (!data.response) {
            throw new Error('Invalid response format: missing response field');
        }
        
        // Remove processing indicator
        removeProcessingIndicator(processingId);
        
        // Add bot response to chat with tools info
        addMessageToChat(data.response, 'bot', data.sources, data.tools_used);
        
    } catch (error) {
        console.error('Error sending message:', error);
        removeProcessingIndicator(processingId);
        addMessageToChat('Sorry, I encountered an error while processing your request. Please try again.', 'bot');
        showToast('Failed to send message: ' + error.message, 'error');
    } finally {
        isLoading = false;
        toggleSendButton();
        toggleAddButton();
    }
}

function addMessageToChat(text, sender, sources = null, tools = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;

    contentDiv.appendChild(textDiv);

    // Add tools used if provided (for bot messages)
    if (tools && tools.length > 0 && sender === 'bot') {
        const toolsDiv = document.createElement('div');
        toolsDiv.className = 'message-tools';
        
        const toolsTitle = document.createElement('div');
        toolsTitle.className = 'tools-title';
        toolsTitle.innerHTML = '<i class="fas fa-tools"></i> Tools Used:';
        toolsDiv.appendChild(toolsTitle);

        const toolsList = document.createElement('div');
        toolsList.className = 'tools-list';
        
        tools.forEach(tool => {
            const toolItem = document.createElement('span');
            toolItem.className = 'tool-item';
            toolItem.textContent = tool;
            toolsList.appendChild(toolItem);
        });
        
        toolsDiv.appendChild(toolsList);
        contentDiv.appendChild(toolsDiv);
    }

    // Add collapsible sources if provided
    if (sources && sources.length > 0) {
        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'sources-container';
        
        const sourcesToggle = document.createElement('div');
        sourcesToggle.className = 'sources-toggle';
        sourcesToggle.innerHTML = `
            <i class="fas fa-chevron-right"></i>
            <span>Sources (${sources.length})</span>
        `;
        
        const sourcesContent = document.createElement('div');
        sourcesContent.className = 'sources-content collapsed';

        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            sourceItem.innerHTML = `
                <h4>${source.title}</h4>
                <a href="${source.url}" target="_blank" rel="noopener noreferrer">${source.url}</a>
                <span class="source-score">Score: ${(source.relevance_score * 100).toFixed(0)}%</span>
            `;
            
            sourcesContent.appendChild(sourceItem);
        });

        // Toggle functionality
        sourcesToggle.addEventListener('click', () => {
            const icon = sourcesToggle.querySelector('i');
            const isCollapsed = sourcesContent.classList.contains('collapsed');
            
            if (isCollapsed) {
                sourcesContent.classList.remove('collapsed');
                icon.classList.remove('fa-chevron-right');
                icon.classList.add('fa-chevron-down');
            } else {
                sourcesContent.classList.add('collapsed');
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-right');
            }
        });
        
        sourcesContainer.appendChild(sourcesToggle);
        sourcesContainer.appendChild(sourcesContent);
        contentDiv.appendChild(sourcesContainer);
    }

    if (sender === 'user') {
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(avatarDiv);
    } else {
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
    }

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function addArticle() {
    const url = articleUrl.value.trim();
    if (!url || !isValidUrl(url) || isLoading) return;

    setLoading(true);
    showArticleStatus('Adding article...', 'loading');

    try {
        const response = await fetch(`${API_BASE_URL}${ENDPOINTS.ingest}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                urls: [url],
                batch_size: 1
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.results.successful > 0) {
            showArticleStatus('Article added successfully!', 'success');
            articleUrl.value = '';
            toggleAddButton();
            showToast('Article added to database', 'success');
            await updateArticleCount();
        } else {
            showArticleStatus('Failed to add article. Please check the URL.', 'error');
            showToast('Failed to add article', 'error');
        }
        
    } catch (error) {
        console.error('Error adding article:', error);
        showArticleStatus('Failed to add article. Please try again.', 'error');
        showToast('Failed to add article', 'error');
    } finally {
        setLoading(false);
        setTimeout(() => {
            hideArticleStatus();
        }, 5000);
    }
}

function showArticleStatus(message, type) {
    articleStatus.textContent = message;
    articleStatus.className = `article-status ${type}`;
}

function hideArticleStatus() {
    articleStatus.style.display = 'none';
}

function addProcessingIndicator() {
    const processingDiv = document.createElement('div');
    processingDiv.className = 'message bot-message processing-message';
    processingDiv.id = 'processing-' + Date.now();

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const processingContent = document.createElement('div');
    processingContent.className = 'processing-content';
    processingContent.innerHTML = `
        <div class="processing-spinner">
            <i class="fas fa-spinner fa-spin"></i>
        </div>
        <span>Processing your request...</span>
    `;

    contentDiv.appendChild(processingContent);
    processingDiv.appendChild(avatarDiv);
    processingDiv.appendChild(contentDiv);

    chatContainer.appendChild(processingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return processingDiv.id;
}

function removeProcessingIndicator(processingId) {
    const processingElement = document.getElementById(processingId);
    if (processingElement) {
        processingElement.remove();
    }
}

function setLoading(loading) {
    isLoading = loading;
    loadingOverlay.style.display = loading ? 'flex' : 'none';
    toggleSendButton();
    toggleAddButton();
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

// Utility function to format text for better display
function formatText(text) {
    // Basic text formatting - can be enhanced
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

// Error handling for fetch requests
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showToast('An unexpected error occurred', 'error');
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message from anywhere
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (chatInput.value.trim()) {
            sendMessage();
        }
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        chatInput.value = '';
        articleUrl.value = '';
        updateCharCount();
        toggleSendButton();
        toggleAddButton();
    }
});

// Auto-focus chat input when page loads
setTimeout(() => {
    chatInput.focus();
}, 500);
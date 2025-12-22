// src\components\Chat\ChatPanel.tsx
import React, { useState, useRef, useEffect, useContext } from 'react';
import { FiSend, FiX, FiMessageSquare, FiTrash2 } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import { useHistory } from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { AuthContext } from '@site/src/components/AuthContext';
import styles from './ChatPanel.module.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  redirect?: string; // Optional redirect URL
  redirectStatus?: 'pending' | 'navigating' | 'done'; // Redirect status for UI
  isThinking?: boolean;
}

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  selectedText?: string;
}

type PanelSize = 'small' | 'medium' | 'large';


// -------------------------------------------------------------------------
// API Configuration
let API_URL = 'https://simple-hackathon-physical-ai-and-humanoid-roboti-production.up.railway.app' ;

// Agar browser mein 'localhost' likha hai, to Local Backend use karo
if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
  API_URL = 'http://localhost:8000';
}

const API_KEY = 'password123';
// -------------------------------------------------------------------------



// Regex to detect redirect commands in response
const REDIRECT_REGEX = /\[\[REDIRECT:([^\]]+)\]\]/;

// Session storage key for chat messages
const CHAT_MESSAGES_KEY = 'physical-ai-chat-messages';

// Helper function to make API request with timeout and proper mobile support
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number = 60000 // 60 second timeout for mobile networks
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      mode: 'cors', // Explicit CORS mode for mobile browsers
      credentials: 'omit', // Don't send credentials for CORS
    });
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    
    return response;
  } catch (err) {
    clearTimeout(timeoutId);
    
    if (err instanceof Error) {
      if (err.name === 'AbortError') {
        throw new Error('Request timed out. Please check your connection and try again.');
      }
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        throw new Error('Network error. Please check your internet connection.');
      }
    }
    throw err;
  }
}

export default function ChatPanel({ isOpen, onClose, selectedText }: ChatPanelProps): React.JSX.Element {
  const { user, isAuthenticated } = useContext(AuthContext);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [panelSize, setPanelSize] = useState<PanelSize>('medium');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const routerHistory = useHistory();
  
  // Get base URL for proper routing
  const baseUrl = useBaseUrl('/');

  // Load messages from session storage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    try {
      const savedMessages = sessionStorage.getItem(CHAT_MESSAGES_KEY);
      if (savedMessages) {
        const parsed = JSON.parse(savedMessages);
        // Convert timestamp strings back to Date objects and clear redirect status
        const restored = parsed.map((m: Message) => ({
          ...m,
          timestamp: new Date(m.timestamp),
          redirectStatus: m.redirectStatus === 'navigating' ? 'done' : m.redirectStatus,
        }));
        setMessages(restored);
      }
    } catch (e) {
      console.error('Failed to restore chat messages:', e);
    }
  }, []);

  // Save messages to session storage when they change
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    try {
      sessionStorage.setItem(CHAT_MESSAGES_KEY, JSON.stringify(messages));
    } catch (e) {
      console.error('Failed to save chat messages:', e);
    }
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    const thinkingMessage: Message = {
      id: `thinking-${Date.now()}`,
      role: 'assistant',
      content: 'Thinking...',
      timestamp: new Date(),
      isThinking: true,
    };

    setMessages(prev => [...prev, userMessage, thinkingMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetchWithTimeout(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json', 
          'X-API-Key': API_KEY 
        },
        body: JSON.stringify({
          message: input,
          history: messages
            .filter(m => !m.isThinking)
            .map(m => ({
              user_message: m.role === 'user' ? m.content : '',
              ai_response: m.role === 'assistant' ? m.content : '',
            }))
            .filter(h => h.user_message || h.ai_response),
          selected_text: selectedText || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Replace thinking message with actual response
      setMessages(prev => 
        prev.map(msg => 
          msg.id === thinkingMessage.id 
            ? {
                ...msg,
                id: Date.now().toString(),
                content: data.response,
                isThinking: false,
              }
            : msg
        )
      );

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      // Remove the thinking message on error
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id));
    } finally {
      setIsLoading(false);
    }
  };
  
  const retryLastMessage = () => {
    // Find the last user message and retry
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMessage) {
      setError(null);
      setInput(lastUserMessage.content);
      // Remove the last user message so it can be re-sent
      setMessages(prev => prev.filter(m => m.id !== lastUserMessage.id));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
    // Also clear from session storage
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem(CHAT_MESSAGES_KEY);
    }
  };

  if (!isOpen) return null;

  return (
    <div className={`${styles.chatPanel} ${styles[panelSize]}`}>
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          <FiMessageSquare />
          <span>AI Assistant</span>
        </div>
        <div className={styles.headerControls}>
          <button 
            onClick={() => setPanelSize('small')} 
            className={`${styles.sizeButton} ${panelSize === 'small' ? styles.active : ''}`}
            title="Small view"
          >
            S
          </button>
          <button 
            onClick={() => setPanelSize('medium')} 
            className={`${styles.sizeButton} ${panelSize === 'medium' ? styles.active : ''}`}
            title="Medium view"
          >
            M
          </button>
          <button 
            onClick={() => setPanelSize('large')} 
            className={`${styles.sizeButton} ${panelSize === 'large' ? styles.active : ''}`}
            title="Large view"
          >
            L
          </button>
          <button 
            onClick={clearChat} 
            className={styles.clearButton} 
            aria-label="Clear chat"
            title="Clear chat history"
            disabled={messages.length === 0}
          >
            <FiTrash2 size={14} />
          </button>
          <button onClick={onClose} className={styles.closeButton} aria-label="Close chat">
            <FiX />
          </button>
        </div>
      </div>

      {selectedText && (
        <div className={styles.selectedTextBanner}>
          <strong>Context:</strong> {selectedText.slice(0, 100)}...
        </div>
      )}

      <div className={styles.messages}>
        {messages.length === 0 && (
          <div className={styles.welcomeMessage}>
            <h3>Welcome to the Physical AI Assistant!</h3>
            <p>Ask me anything about ROS 2, Gazebo, NVIDIA Isaac, or humanoid robotics.</p>
            <div style={{ fontSize: '0.85rem', marginTop: '12px', color: 'var(--ifm-color-emphasis-600)' }}>
              <p style={{ marginBottom: '6px' }}><strong>💡 Tips:</strong></p>
              <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                <li>Ask for "detailed" or "in-depth" explanations for comprehensive answers</li>
                <li>Select text on the page for context-aware help</li>
                <li><strong>🔗 Navigation:</strong> Ask me to find specific content (e.g., "take me to VSLAM", "find kinematics")</li>
                <li><strong>📍 Highlighting:</strong> I'll highlight the exact section when I redirect you!</li>
              </ul>
            </div>
          </div>
        )}
        
        {messages.map(message => (
          <div
            key={message.id}
            className={`${styles.message} ${styles[message.role]} ${message.redirect ? styles.redirectMessage : ''}`}
          >
            <div className={styles.messageContent}>
              {message.role === 'assistant' ? (
                <>
                  <ReactMarkdown>
                    {message.content || (isLoading ? 'Thinking...' : '')}
                  </ReactMarkdown>
                  {message.redirect && (
                    <div className={`${styles.redirectIndicator} ${message.redirectStatus === 'navigating' ? styles.navigating : ''}`}>
                      {message.redirectStatus === 'navigating' ? (
                        <>
                          <span className={styles.redirectSpinner}>⏳</span>
                          <span>Taking you there now...</span>
                        </>
                      ) : message.redirectStatus === 'done' ? (
                        <>
                          <span className={styles.redirectIcon}>✅</span>
                          <span>Redirected!</span>
                        </>
                      ) : (
                        <>
                          <span className={styles.redirectIcon}>🔗</span>
                          <span>Will redirect to: {message.redirect}</span>
                        </>
                      )}
                    </div>
                  )}
                </>
              ) : (
                message.content
              )}
            </div>
          </div>
        ))}
        
        {error && (
          <div className={styles.error}>
            <span>{error}</span>
            <div className={styles.errorButtons}>
              <button onClick={retryLastMessage}>Retry</button>
              <button onClick={() => setError(null)}>Dismiss</button>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className={styles.inputArea}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about Physical AI..."
          className={styles.input}
          rows={1}
          disabled={isLoading}
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim() || isLoading}
          className={styles.sendButton}
          aria-label="Send message"
        >
          <FiSend />
        </button>
      </div>
    </div>
  );
}

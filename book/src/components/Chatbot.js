import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';
import { chatApi } from './ChatbotAPI';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [useStrictContext, setUseStrictContext] = useState(false);
  const messagesEndRef = useRef(null);

  // Function to get selected text from the page
  const getSelectedText = () => {
    const selectedText = window.getSelection?.toString()?.trim() || '';
    setSelectedText(selectedText);
    return selectedText;
  };

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message
  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = { role: 'user', content: inputMessage };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputMessage('');
    setIsLoading(true);

    try {
      let response;

      if (useStrictContext && selectedText) {
        // Use strict context mode with selected text
        response = await chatApi.chatSelectedText(
          inputMessage,
          selectedText,
          newMessages.filter(msg => msg.role !== 'assistant').map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        );
      } else {
        // Use full book RAG mode
        response = await chatApi.chatFullBook(
          inputMessage,
          newMessages.filter(msg => msg.role !== 'assistant').map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        );
      }

      const botMessage = {
        role: 'assistant',
        content: response.response,
        sources: response.sources || []
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Toggle chat window
  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      getSelectedText(); // Capture any selected text when opening
    }
  };

  // Clear chat history
  const clearChat = () => {
    setMessages([]);
    setInputMessage('');
  };

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : ''}`}>
      {/* Floating button to open chat */}
      {!isOpen && (
        <button className="chatbot-toggle-btn" onClick={toggleChat}>
          💬 AI Assistant
        </button>
      )}

      {isOpen && (
        <div className="chatbot-window">
          {/* Header */}
          <div className="chatbot-header">
            <h3>Book AI Assistant</h3>
            <div className="header-controls">
              <label className="strict-mode-toggle">
                <input
                  type="checkbox"
                  checked={useStrictContext}
                  onChange={(e) => setUseStrictContext(e.target.checked)}
                />
                Strict Context Mode
              </label>
              <button className="clear-chat-btn" onClick={clearChat}>Clear</button>
              <button className="close-chat-btn" onClick={toggleChat}>×</button>
            </div>
          </div>

          {/* Selected text indicator */}
          {useStrictContext && selectedText && (
            <div className="selected-text-preview">
              <strong>Selected text:</strong> "{selectedText.substring(0, 100)}..."
            </div>
          )}
          
          {useStrictContext && !selectedText && (
            <div className="selected-text-warning">
              No text selected. Select text on the page to use strict context mode.
            </div>
          )}

          {/* Messages container */}
          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics book.</p>
                <p>You can ask me anything about the book content!</p>
                <p>In strict context mode, I'll only answer based on the text you've selected.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-content">
                    {message.content}
                    {message.sources && message.sources.length > 0 && (
                      <div className="sources">
                        <details>
                          <summary>Sources</summary>
                          <ul>
                            {message.sources.slice(0, 3).map((source, idx) => (
                              <li key={idx}>
                                {source.file_path ? `${source.title} (${source.file_path})` : source.text}
                              </li>
                            ))}
                          </ul>
                        </details>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="chatbot-input-area">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={useStrictContext && !selectedText 
                ? "Select text on the page first, then type your question..." 
                : "Ask a question about the book content..."}
              rows="2"
              disabled={isLoading}
              className={useStrictContext && !selectedText ? "disabled-input" : ""}
            />
            <button 
              onClick={sendMessage} 
              disabled={!inputMessage.trim() || isLoading || (useStrictContext && !selectedText)}
              className="send-button"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
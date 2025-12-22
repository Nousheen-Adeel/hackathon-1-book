import React, { useState } from 'react';
import { BrowserOnly } from '@docusaurus/core';

const ChatbotComponent = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (res.ok) {
        const data = await res.json();
        setResponse(data.response);
      } else {
        setResponse('Error: Could not get response from the server.');
      }
    } catch (error) {
      setResponse('Error: Network issue occurred while connecting to the server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chatbot-container" style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h3>Robotics AI Chatbot</h3>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '10px' }}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about robotics..."
            style={{
              width: '70%',
              padding: '10px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              marginRight: '10px'
            }}
            disabled={isLoading}
          />
          <button 
            type="submit" 
            style={{
              padding: '10px 15px',
              backgroundColor: '#007cba',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isLoading ? 'not-allowed' : 'pointer'
            }}
            disabled={isLoading}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </form>
      {response && (
        <div style={{ 
          marginTop: '15px', 
          padding: '10px', 
          backgroundColor: '#f0f0f0', 
          borderRadius: '4px',
          minHeight: '20px'
        }}>
          <strong>Response:</strong> {response}
        </div>
      )}
    </div>
  );
};

// Wrapper component that renders only in browser environment
const Chatbot = () => {
  return (
    <BrowserOnly fallback={<div>Loading chatbot...</div>}>
      {() => <ChatbotComponent />}
    </BrowserOnly>
  );
};

export default Chatbot;
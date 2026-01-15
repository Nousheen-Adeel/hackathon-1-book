import React, { useState, useEffect } from 'react';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

const ChatbotIntegration = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const { siteConfig } = useDocusaurusContext();

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  const closeChatbot = () => {
    setIsOpen(false);
  };

  // Load the chatbot script dynamically when component mounts
  useEffect(() => {
    if (!isLoaded) {
      setIsLoaded(true);
    }
  }, [isLoaded]);

  return (
    <>
      <Head>
        <script>
          {`
            // Chatbot script logic would go here if needed
          `}
        </script>
      </Head>
      
      <div id="chatbot-container">
        <div id="chatbot-panel" className={isOpen ? 'open' : ''}>
          <div id="chatbot-header">
            <h3 id="chatbot-title">AI Assistant</h3>
            <button id="chatbot-close" onClick={closeChatbot} aria-label="Close chatbot">
              ×
            </button>
          </div>
          <iframe
            id="chatbot-iframe"
            src="/../../rag_frontend/index.html"
            title="AI Textbook Assistant"
            allow="microphone"
            style={{ width: '100%', height: '400px', border: 'none' }}
          ></iframe>
        </div>
        <button
          id="chatbot-toggle"
          onClick={toggleChatbot}
          aria-label={isOpen ? "Close AI Assistant" : "Open AI Assistant"}
          aria-expanded={isOpen}
        >
          💬
        </button>
      </div>
    </>
  );
};

export default ChatbotIntegration;
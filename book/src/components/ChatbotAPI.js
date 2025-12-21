// API utility functions for the chatbot
// Using the backend server
const API_BASE = 'http://localhost:8000';

export const chatApi = {
  async chatFullBook(message, history = [], temperature = 0.7) {
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          history,
          temperature
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in chatFullBook:', error);
      throw error;
    }
  },

  async chatSelectedText(message, selectedText, history = [], temperature = 0.7) {
    try {
      const response = await fetch(`${API_BASE}/chat/selected`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          selected_text: selectedText,
          history,
          temperature
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error in chatSelectedText:', error);
      throw error;
    }
  }
};
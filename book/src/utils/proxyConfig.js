// Development server configuration with proxy
const path = require('path');

// Proxy configuration for development
const proxyConfig = {
  '/api/**': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    pathRewrite: {
      '^/api': '', // Remove /api prefix when forwarding
    },
  },
};

module.exports = {
  proxyConfig
};
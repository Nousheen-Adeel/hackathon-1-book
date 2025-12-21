# Physical AI & Humanoid Robotics Book

A comprehensive guide to embodied intelligence and humanoid robotics with integrated AI assistant, built with Docusaurus.

## Features

- Modern, responsive design with indigo/cyan color scheme
- AI-powered chatbot assistant integrated into the interface
- Optimized for robotics and AI education
- Dark/light mode support
- Mobile-friendly layout

## About

This book provides a comprehensive, hands-on guide to embodied intelligence and humanoid robotics. It covers fundamental concepts through advanced topics like reinforcement learning, vision-language-action models, and human-robot interaction.

## Getting Started

### Prerequisites
- Node.js 18 or higher
- Python 3.11
- The RAG backend server running on `http://localhost:8000`

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/physical-ai-humanoid-robotics.git
cd physical-ai-humanoid-robotics

# Install dependencies
npm install
```

### Local Development

```bash
# Start the RAG backend server in a separate terminal
cd ../rag_backend
pip install -r requirements.txt
python main.py

# In another terminal, start the Docusaurus frontend
cd ../book
npm start

# The site will be accessible at http://localhost:3000
```

### Development with Both Servers

To run both servers simultaneously during development:
```bash
npm run start:dev
```

### Build for Production

```bash
# Build the static site
npm run build

# Serve the built site locally for testing
npm run serve
```

## Design

- **Color Scheme**: Deep indigo primary (#2C3E50) with cyan accent (#00BCD4)
- **Typography**: Inter font for clean, readable text
- **Layout**: Clean sidebar navigation with readable content width
- **Accessibility**: High contrast for readability and proper semantic structure

## AI Assistant

The book includes an integrated RAG (Retrieval-Augmented Generation) chatbot that can answer questions about the textbook content:

- Access via the floating chat button on any page
- Answers based strictly on book content
- Source attribution for all responses
- Two modes: general book queries and strict context mode

## Project Structure

```
book/
├── docs/                 # Book content organized by parts and chapters
│   ├── part-i-foundations/
│   ├── part-ii-perception/
│   ├── part-iii-motion/
│   ├── part-iv-intelligence/
│   └── part-v-integration/
├── src/                  # Custom components and styling
│   ├── components/       # Chatbot and custom UI components
│   ├── css/              # Custom styling (custom.css)
│   └── pages/
├── notebooks/            # Jupyter notebooks for interactive examples
├── diagrams/             # Diagram source files
├── assets/               # Images and other assets
├── docusaurus.config.js  # Docusaurus configuration
├── sidebars.js           # Navigation sidebar configuration
└── package.json          # Dependencies and scripts
```

## Customization

- Theme colors and styles are in `src/css/custom.css`
- Chatbot UI is in `src/components/Chatbot.js` and `src/components/Chatbot.css`
- Layout customization is in `src/theme/Layout.js`

## Contributing

This book is designed for learning, but contributions are welcome:

1. Fork the repository
2. Create a feature branch for your changes
3. Add your content following the established structure
4. Submit a pull request with a clear description of your changes

## License

This book is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
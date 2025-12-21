# Physical AI & Humanoid Robotics Book Quickstart Guide

## Getting Started

This guide will help you set up the development environment for writing and building the Physical AI & Humanoid Robotics book.

### Prerequisites

- Node.js (v16 or higher)
- Git
- Python 3.8+ (for code sample testing)
- ROS 2 Humble Hawksbill (for testing examples)
- Basic familiarity with Markdown

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install Docusaurus**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run start
   ```
   This will start a local server at `http://localhost:3000` with live reloading.

### Project Structure

```
book/
├── docs/               # Chapter content
│   ├── part-i-foundations/
│   ├── part-ii-perception/
│   ├── part-iii-motion/
│   ├── part-iv-intelligence/
│   └── part-v-integration/
├── src/                # Custom components and styling
├── notebooks/          # Jupyter notebooks for code samples
├── diagrams/           # Diagram source files
├── assets/             # Images and other assets
├── docusaurus.config.js # Docusaurus configuration
└── package.json        # Dependencies and scripts
```

### Writing a New Chapter

1. **Create the chapter directory** in the appropriate part:
   ```bash
   mkdir docs/part-i-foundations/chapter-x-topic-name
   ```

2. **Add the chapter content** as `index.md`:
   ```markdown
   ---
   title: Chapter X: Topic Name
   sidebar_position: X
   ---

   # Chapter X: Topic Name

   ## Learning Goals
   - Goal 1
   - Goal 2

   ## Key Technologies
   - Technology 1
   - Technology 2

   ## Content...
   ```

3. **Update the sidebar configuration** in the appropriate `_category_.json` file.

### Adding Code Samples

1. **For simple code snippets**, use standard Markdown syntax:
   ```python
   # Python code example
   import rospy
   from std_msgs.msg import String
   ```

2. **For complex examples**, create Jupyter notebooks in the `notebooks/` directory and reference them in the chapter.

### Adding Diagrams

1. **Create diagram source files** in the `diagrams/` directory
2. **Export as images** to the `assets/` directory
3. **Reference in markdown**:
   ```markdown
   ![Diagram Description](/assets/diagram-name.png)
   ```

### Building for Production

```bash
npm run build
```

This creates an optimized build in the `build/` directory that can be served statically.

### Deploying to GitHub Pages

The deployment happens automatically via GitHub Actions when changes are pushed to the main branch. Ensure your `docusaurus.config.js` is properly configured for your GitHub Pages URL.
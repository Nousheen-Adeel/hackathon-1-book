FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Build the Docusaurus site
RUN npm run build

# Expose the port
EXPOSE 3000

# Serve the built site using the built-in server
CMD ["sh", "-c", "npx serve -l $PORT ./build"]
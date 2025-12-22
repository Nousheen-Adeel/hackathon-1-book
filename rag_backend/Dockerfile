FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (though Railway will override this with the PORT env var)
EXPOSE $PORT

# Run the application
CMD ["python", "start_server.py"]
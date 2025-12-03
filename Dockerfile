# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# Removed 'software-properties-common' to fix the build error
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (to cache dependencies)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run expects the app to listen on port 8080 by default
EXPOSE 8080

# Command to run the Streamlit app on port 8080
CMD ["streamlit", "run", "rag_streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
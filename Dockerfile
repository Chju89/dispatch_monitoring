# --- Base Image ---
FROM python:3.10-slim

# --- Set working directory ---
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Copy project files ---
COPY . .

# --- Install Python dependencies ---
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Expose port for Streamlit ---
EXPOSE 8501

# --- Default command to run app ---
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


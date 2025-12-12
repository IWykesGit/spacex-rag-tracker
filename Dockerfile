# Start from slim Python base (fast, secure)
FROM python:3.11-slim 

# Set working directory in container
WORKDIR /app

# Copy this first to cache 
COPY requirements.txt .

# Stops HF from downloading models during pip install
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Install dependencies (no caching to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# copy everything
COPY . .

# Install pytest and run tests during build
RUN pip install --no-cache-dir pytest-html && pytest --html=/app/report.html --self-contained-html -v --disable-warnings

#Tell Docker about the port
EXPOSE 8000

# Command to run the app when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
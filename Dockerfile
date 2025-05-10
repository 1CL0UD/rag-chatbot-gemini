# Use official Python 3.13 image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (faster alternative to pip)
RUN pip install uv && \
    uv pip install -r pyproject.toml

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=7860
ENV SHARE=false

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "main.py"]
# RAG Chatbot with Gradio and Gemini

A Retrieval-Augmented Generation (RAG) chatbot built with Gradio and Google's Gemini model. This chatbot can answer questions based on your uploaded documents, providing context-aware responses.

## Features

- **Document Processing**: Upload PDF and text files to create a knowledge base
- **Vector Search**: Uses ChromaDB to find relevant information from your documents
- **Environment Configuration**: Easily configure API keys and settings via .env file
- **User-friendly Interface**: Two-tab interface for setup and chat

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the provided example:

   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your Gemini API key and preferred settings:
   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-pro
   DEFAULT_TEMPERATURE=0.7
   PORT=7860
   SHARE=false
   ```

## Usage

1. Run the application:

   ```bash
   python main.py
   ```

2. Open your browser and navigate to http://localhost:7860 (or the port you specified)

3. In the Setup tab:

   - Validate your API key
   - Upload documents
   - Configure model and temperature settings
   - Process the documents

4. Switch to the Chat tab and start asking questions about your documents

## Configuration Options

You can configure the following options in your `.env` file:

| Variable              | Description                          | Default          |
| --------------------- | ------------------------------------ | ---------------- |
| `GEMINI_API_KEY`      | Your Google Gemini API key           | None             |
| `GEMINI_MODEL`        | The Gemini model to use              | gemini-2.0-flash |
| `DEFAULT_TEMPERATURE` | Control randomness (0.0-1.0)         | 0.7              |
| `PORT`                | Server port for the Gradio interface | 7860             |
| `SHARE`               | Whether to create a public link      | false            |

## Project Structure

- `main.py` - Main application file
- `config.py` - Configuration handling
- `requirements.txt` - Required Python packages
- `.env` - Environment variables (create this from .env.example)
- `.env.example` - Example environment variables file

## How It Works

1. **Document Processing**:

   - Documents are split into chunks of 1000 characters with 200 character overlap
   - Each chunk is embedded using Google's embedding model
   - The embeddings are stored in a ChromaDB vector database

2. **Query Processing**:
   - When a question is asked, it's embedded using the same model
   - Vector similarity search finds the most relevant document chunks
   - The chunks are combined into a prompt with the question
   - Gemini generates a response based on the provided context

## Getting an API Key

To get a Google Gemini API key:

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an account if needed
3. Generate a new API key
4. Copy the key to your `.env` file

## License

This project is licensed under the MIT License - see the LICENSE file for details.

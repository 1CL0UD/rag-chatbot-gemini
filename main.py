import os
import gradio as gr
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
from config import get_api_key, get_model_settings, get_app_settings

# Initialize Gemini API
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    
    # Verify API key works by getting available models
    try:
        models = genai.list_models()
        available_models = [model.name for model in models if "generateContent" in model.supported_generation_methods]
        if not available_models:
            return False, "API key works but no models support text generation"
        return True, f"API key validated. Available models: {', '.join(available_models)}"
    except Exception as e:
        return False, f"API key validation failed: {str(e)}"

# Document processing
def process_documents(files):
    documents = []
    
    for file in files:
        try:
            # For newer Gradio versions, file might be a dict or object with different structure
            if isinstance(file, dict) and "path" in file:
                file_path = file["path"]
                file_name = file.get("name", "")
            elif hasattr(file, "name") and hasattr(file, "path"):
                file_path = file.path
                file_name = file.name
            else:
                # Create a temporary file for older Gradio versions
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                file_path = temp_file.name
                
                # Get file content - handle different Gradio file object structures
                if hasattr(file, "read"):
                    # File-like object
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    file_name = getattr(file, "name", "unknown")
                else:
                    # May be a string path or other format
                    file_path = str(file)
                    file_name = os.path.basename(file_path)
            
            # Handle different file types
            if file_name.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:  # Default to text loader
                loader = TextLoader(file_path)
                
            documents.extend(loader.load())
            
            # Clean up if we created a temp file
            if 'temp_file' in locals():
                os.unlink(file_path)
                
        except Exception as e:
            print(f"Error loading file: {e}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks

# Create vector store
def create_vector_store(api_key, chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Create vector store from documents
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vector_store

# Generate response from Gemini
def generate_response(api_key, query, vector_store, k=3, temperature=0.7, model="gemini-2.0-flash"):
    # Get relevant documents
    if vector_store:
        docs = vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        context = ""
    
    # Create the prompt
    if context:
        prompt = f"""
        Answer the question based on the following context:
        
        Context:
        {context}
        
        Question: {query}
        
        If the context doesn't contain relevant information to answer the question, 
        please say "I don't have enough information to answer this question."
        """
    else:
        prompt = f"""
        Question: {query}
        
        Please note that I don't have any specific context to draw upon for answering this question.
        """
    
    # Configure the model
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    # Generate response
    try:
        model = genai.GenerativeModel(model_name=model)
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Gradio Interface
def create_chatbot_interface():
    # Get settings from config
    model_settings = get_model_settings()
    default_api_key = get_api_key() or ""
    
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Chatbot with Gemini")
        
        with gr.Tab("Setup"):
            api_key_input = gr.Textbox(
                label="Enter Gemini API Key", 
                type="password",
                value=default_api_key
            )
            validate_button = gr.Button("Validate API Key")
            validation_output = gr.Markdown()
            
            gr.Markdown("### Upload Documents")
            file_output = gr.File(label="Upload Documents", file_count="multiple")
            model_choice = gr.Dropdown(
                choices=["gemini-2.0-flash"],
                label="Model",
                value=model_settings["model"]
            )
            temperature_slider = gr.Slider(
                minimum=0, 
                maximum=1, 
                value=model_settings["temperature"], 
                label="Temperature"
            )
            process_button = gr.Button("Process Documents")
            processing_output = gr.Markdown()
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Conversation")
            msg = gr.Textbox(label="Message")
            send = gr.Button("Send")
            clear = gr.Button("Clear")
        
        # State variables
        vector_store_state = gr.State(None)
        api_key_state = gr.State(default_api_key if default_api_key else None)
        model_state = gr.State(model_settings["model"])
        temperature_state = gr.State(model_settings["temperature"])
        
        # Setup functions
        def validate_api_key(api_key):
            success, message = initialize_gemini(api_key)
            if success:
                return message, api_key
            else:
                return message, None
        
        validate_button.click(
            validate_api_key, 
            inputs=[api_key_input], 
            outputs=[validation_output, api_key_state]
        )
        
        def process_docs(api_key, files, model, temperature):
            if not api_key:
                return "Please validate your API key first.", None, model, temperature
            
            if not files:
                return "Please upload at least one document.", None, model, temperature
            
            try:
                print(f"Processing files: {files}")
                chunks = process_documents(files)
                if not chunks:
                    return "No content could be extracted from the documents.", None, model, temperature
                
                vector_store = create_vector_store(api_key, chunks)
                return f"Processed {len(chunks)} chunks from {len(files)} files. Ready to chat!", vector_store, model, temperature
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error processing documents: {e}")
                print(f"Error details: {error_details}")
                return f"Error processing documents: {str(e)}", None, model, temperature
        
        process_button.click(
            process_docs,
            inputs=[api_key_state, file_output, model_choice, temperature_slider],
            outputs=[processing_output, vector_store_state, model_state, temperature_state]
        )
        
        # Chat functions
        def user_message(user_message, history, api_key, vector_store, model, temperature):
            if not api_key:
                return "", history + [[user_message, "Please validate your API key in the Setup tab."]]
            
            if not vector_store and user_message.strip() != "":
                return "", history + [[user_message, "Please process some documents in the Setup tab before chatting."]]
            
            return "", history + [[user_message, None]]
        
        def bot_response(history, api_key, vector_store, model, temperature):
            if history and history[-1][1] is None:
                query = history[-1][0]
                bot_message = generate_response(api_key, query, vector_store, k=3, temperature=temperature, model=model)
                history[-1][1] = bot_message
            return history
        
        msg.submit(
            user_message,
            inputs=[msg, chatbot, api_key_state, vector_store_state, model_state, temperature_state],
            outputs=[msg, chatbot]
        ).then(
            bot_response,
            inputs=[chatbot, api_key_state, vector_store_state, model_state, temperature_state],
            outputs=[chatbot]
        )
        
        send.click(
            user_message,
            inputs=[msg, chatbot, api_key_state, vector_store_state, model_state, temperature_state],
            outputs=[msg, chatbot]
        ).then(
            bot_response,
            inputs=[chatbot, api_key_state, vector_store_state, model_state, temperature_state],
            outputs=[chatbot]
        )
        
        def clear_chat():
            return []
        
        clear.click(clear_chat, outputs=[chatbot])
        
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_chatbot_interface()
    
    # Get app settings
    app_settings = get_app_settings()
    
    demo.launch(server_port=app_settings["port"], share=app_settings["share"])
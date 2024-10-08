import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any, Literal
import logging
import tempfile
import json

# Load environment variables from .env file
load_dotenv()

# Constants
SYSTEM_PROMPT_FILE = "system_prompt.txt"
EXTERNAL_DOC_FILE = "external_doc.txt"
DEFAULT_MODEL = "gpt-4o"
TTS_MODELS = ["tts-1", "tts-1-hd"]
VOICE_OPTIONS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_system_prompt(file_path: str) -> str:
    """
    Load the system prompt from a text file.

    Args:
        file_path (str): Path to the system prompt file.

    Returns:
        str: The system prompt.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        st.error(f"System prompt file '{file_path}' not found.")
        return ""

def load_external_document(file_path: str) -> str:
    """
    Load external document content from a text file.

    Args:
        file_path (str): Path to the external document file.

    Returns:
        str: The content of the external document.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        st.error(f"External document file '{file_path}' not found.")
        return ""

def initialize_session(state: Any, system_prompt: str) -> None:
    """
    Initialize session state variables.

    Args:
        state (Any): Streamlit session state.
        system_prompt (str): The system prompt to initialize messages.
    """
    if "messages" not in state:
        state.messages = []
    if "system_prompt" not in state:
        state.system_prompt = system_prompt
    if "openai_model" not in state:
        state.openai_model = DEFAULT_MODEL
    if "tts_data" not in state:
        state.tts_data = {}  # Stores audio bytes indexed by message index

def display_chat_history(messages: List[Dict[str, str]], tts_data: Dict[int, bytes]) -> None:
    """Display the chat history in the Streamlit app with TTS options."""
    for idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display existing audio if available
            if message["role"] == "assistant" and idx in tts_data:
                st.audio(tts_data[idx], format="audio/mp3", start_time=0)
        
        # If the message is from the assistant, display TTS options
        if message["role"] == "assistant":
            with st.expander("🔊 Text-to-Speech Options"):
                # Unique keys for each assistant message's TTS
                voice_key = f"voice_{idx}"
                model_key = f"model_{idx}"
                button_key = f"button_{idx}"
                
                selected_voice = st.selectbox(
                    "Choose a voice:",
                    options=VOICE_OPTIONS,
                    key=voice_key
                )
                
                selected_model = st.selectbox(
                    "Choose TTS model:",
                    options=TTS_MODELS,
                    key=model_key
                )
                
                generate_button = st.button(
                    "Generate Speech",
                    key=button_key
                )
                
                if generate_button:
                    # Generate speech using TTS
                    with st.spinner("Generating speech..."):
                        tts_response = get_tts_response(selected_model, selected_voice, message["content"])
                    
                    if tts_response:
                        # Store the audio bytes in session state
                        tts_data[idx] = tts_response
                        st.success("Speech generated successfully!")
                        st.audio(tts_response, format="audio/mp3", start_time=0)
                    else:
                        st.error("Failed to generate speech.")

def format_user_message_with_doc(document: str, user_message: str) -> str:
    """Format the user message by embedding the external document within XML delimiters."""
    return f"<content>\n{document}\n</content>\n<user>\n{user_message}\n</user>"

def get_openai_response(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    """Get the assistant's response from OpenAI API and extract the 'response' field."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )
        response = response.choices[0].message.content.strip()
        return response
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        return "Sorry, something went wrong. Please try again."

def get_tts_response(model: Literal["tts-1", "tts-1-hd"], voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"], text: str) -> bytes:
    """Generate speech using OpenAI's TTS model."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY"))
        tts_response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        # Use a temporary file to handle the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_filename = tmp_file.name
        # Write to the temporary file
        tts_response.write_to_file(tmp_filename)
        # Read the audio data into memory
        with open(tmp_filename, "rb") as f:
            audio_bytes = f.read()
        # Clean up the temporary file
        os.remove(tmp_filename)
        return audio_bytes
    except Exception as e:
        logger.error(f"TTS API request failed: {e}")
        return None

def log_api_request(api_messages: List[Dict[str, str]]) -> None:
    """
    Log the full chat history to the terminal.

    Args:
        api_messages (List[Dict[str, str]]): List of messages including system prompt
    """
    logger.info("API Request Chat History:")
    for msg in api_messages:
        logger.info(f"{msg['role'].upper()}:\n{msg['content']}")

def main():
    """
    Main function to run the Streamlit ChatGPT-like application.
    """
    st.set_page_config(page_title="ChatGPT Clone", page_icon="💬")
    st.title("ChatGPT-like Clone")

    # Load system prompt and external document
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    external_document = load_external_document(EXTERNAL_DOC_FILE)

    # Initialize session state
    initialize_session(st.session_state, system_prompt)

    # Initialize OpenAI client
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Display chat history
    display_chat_history(st.session_state.messages, st.session_state.tts_data)

    # Handle user input
    user_input = st.chat_input("What would you like to discuss?")
    if user_input:
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare messages for OpenAI API
        # Include system prompt as the first message
        api_messages = [{"role": "system", "content": st.session_state.system_prompt}]

        # Add all previous messages except the latest user message
        for msg in st.session_state.messages[:-1]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Format only the latest user message with the external document
        formatted_latest_user = format_user_message_with_doc(external_document, st.session_state.messages[-1]["content"])
        api_messages.append({"role": "user", "content": formatted_latest_user})

        # Get assistant response
        with st.chat_message("assistant"):
            response = get_openai_response(client, st.session_state.openai_model, api_messages)
            try:
                response_json = json.loads(response)
                response_str = response_json.get("response")
            except json.JSONDecodeError:
                # If response is not in JSON format, display the raw response
                response_str = response
            st.write(response_str)

        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_str})

        # Log full chat history to terminal
        api_messages.append({"role": "assistant", "content": response})
        log_api_request(api_messages)

if __name__ == "__main__":
    main()
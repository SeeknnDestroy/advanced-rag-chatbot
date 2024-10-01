import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any
import logging
from pprint import pformat


# Load environment variables from .env file
load_dotenv()

# Constants
SYSTEM_PROMPT_FILE = "system_prompt.txt"
EXTERNAL_DOC_FILE = "external_doc.txt"
DEFAULT_MODEL = "gpt-4o-mini"

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

def display_chat_history(messages: List[Dict[str, str]]) -> None:
    """
    Display the chat history in the Streamlit app.

    Args:
        messages (List[Dict[str, str]]): List of messages with roles and content.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def format_user_message_with_doc(document: str, user_message: str) -> str:
    """
    Format the user message by embedding the external document within XML delimiters.

    Args:
        document (str): The external document content.
        user_message (str): The latest user message.

    Returns:
        str: Formatted user message.
    """
    return f"<content>\n{document}\n</content>\n<user>\n{user_message}\n</user>"

def get_openai_response(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    """
    Get the assistant's response from OpenAI API.

    Args:
        client (OpenAI): OpenAI client instance.
        model (str): The model to use for completion.
        messages (List[Dict[str, str]]): List of messages for context.

    Returns:
        str: Assistant's response.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)
        return response
    except Exception as e:
        st.error(f"Error communicating with OpenAI: {e}")
        return "Sorry, I couldn't process that."
    
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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Display chat history
    display_chat_history(st.session_state.messages)

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

        # Log the API request messages
        log_api_request(api_messages)

        # Get assistant response
        with st.chat_message("assistant"):
            response = get_openai_response(client, st.session_state.openai_model, api_messages)

        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
import streamlit as st
import google.generativeai as genai

# --- Configuration ---
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot")

# --- API Key Setup ---
# Option 1: Set via environment variable (recommended for production)
# export GOOGLE_API_KEY="AIzaSyBpJ8ncsHHKwmks1Se9nVUlBm2AEG9yOEA"

# Option 2: Sidebar input (for development)
with st.sidebar:
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    st.markdown("[Get an API key](https://aistudio.google.com/app/apikey)")
    model_name = st.selectbox(
        "Select Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    )

if not api_key:
    st.info("Please enter your Gemini API key in the sidebar to start chatting.")
    st.stop()

# Configure the Gemini API
genai.configure(api_key=api_key)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    model = genai.GenerativeModel(model_name)
    st.session_state.chat = model.start_chat(history=[])

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat.send_message(prompt)
                response_text = response.text
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Sidebar: Clear Chat ---
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        model = genai.GenerativeModel(model_name)
        st.session_state.chat = model.start_chat(history=[])
        st.rerun()
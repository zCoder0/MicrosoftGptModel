import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image

# Load the DialoGPT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    return tokenizer, model

tokenizer, model = load_model()

st.markdown(
    """
    <style>
    .chat-bubble {
        background-color: #f1f0f0;
        color: #333333;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for Navigation


# Main Content
st.title("Welcome to Your Chatbot 🤖")
st.markdown("### A conversational AI experience, built with 💻 Python and 🤗 Transformers.")

# Initialize chat state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

if "conversation" not in st.session_state:
    st.session_state.conversation = ""

# Chat Input and Display
st.write("---")
st.subheader("Chat Here:")
user_input = st.chat_input("Type your message and press Enter!")

if user_input:
    # Process input and generate response
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat(
        [st.session_state.chat_history_ids, new_user_input_ids], dim=-1
    ) if st.session_state.chat_history_ids is not None else new_user_input_ids

    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.6
    )

    bot_response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Append to conversation
    st.session_state.conversation += (
        f"<div class='user-bubble'><b>You:</b> {user_input}</div>"
        f"<div class='chat-bubble'><b>Bot:</b> {bot_response}</div>"
    )

# Display Chat History
st.markdown(f"<div>{st.session_state.conversation}</div>", unsafe_allow_html=True)

# Footer


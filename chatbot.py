import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os



# Set your OpenAI API key securely
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.7,
    model_name="gpt-4o-mini"  
)

# Set up memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Conversation Chain
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Streamlit UI
st.set_page_config(page_title="LLM Chatbot", page_icon=":(")
st.title("LangChain Chatbot")

user_input = st.text_input("You:", key="input")

if user_input:
    response = conversation.predict(input=user_input)
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(response)
    st.write(f"**Bot:** {response}")

# Show history
if st.checkbox("Show Chat History"):
    for message in st.session_state.memory.chat_memory.messages:
        role = "You" if message.type == "human" else "Bot"
        st.markdown(f"**{role}:** {message.content}")

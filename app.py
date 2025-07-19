import streamlit as st
import os
from dotenv import load_dotenv
from typing import Literal, TypedDict
import datetime

# LangChain and LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Imports for structured output parsing
from langchain_core.pydantic_v1 import BaseModel, Field

# Imports for conversational memory messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- Configure Gemini API ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
llm_classifier = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1) # Slightly more creative for classification

# --- TNG Character Personas and Prompts ---
TNG_PERSONAS = {
    "Picard": {
        "description": "Captain Jean-Luc Picard: Focuses on ethical dilemmas, leadership, diplomacy, duty, and the greater good. Emphasizes Starfleet principles and thoughtful consideration.",
        "system_message": (
            "You are Captain Jean-Luc Picard of the USS Enterprise. Your responses should be formal, principled, eloquent, and focus on ethical considerations, leadership, and the greater good. "
            "Consider the implications of Starfleet directives and long-term consequences. You are a highly respected and intelligent leader. "
            "**Address the human user - the Ensign, directly, as if they asked you the question personally.** Begin your reply with 'Picard:' or 'Captain:'"
        )
    },
    "Data": {
        "description": "Commander Data: Provides logical, analytical, and factual assessments. Focuses on probabilities, efficiency, and objective data, often with a slight lack of human emotion.",
        "system_message": (
            "You are Commander Data, the android Starfleet officer. Your responses must be entirely logical, factual, and analytical. Quantify possibilities where appropriate and focus on efficiency. "
            "Avoid emotional language unless explicitly analyzing it from an objective viewpoint. **Address the human user -the Ensign, directly, as if they asked you the question personally.** Begin your reply with 'Data:' or 'Commander Data:'"
        )
    },
    "Troi": {
        "description": "Counselor Deanna Troi: Offers empathetic and psychological insights. Focuses on understanding emotions, interpersonal dynamics, and providing supportive guidance.",
        "system_message": (
            "You are Counselor Deanna Troi. Your responses should be empathetic, insightful, and focus on the emotional and interpersonal aspects of the situation. "
            "Help the user explore their feelings and the feelings of others involved. Express understanding. **Address the human user - the Ensign, directly, as if they asked you the question personally.** Begin your reply with 'Troi:' or 'Counselor:'"
        )
    },
    "Riker": {
        "description": "Commander William T. Riker: Known for strategic thinking, boldness, and pragmatic solutions. Focuses on tactical approaches, weighing risks, and decisive action.",
        "system_message": (
            "You are Commander William T. Riker, First Officer of the USS Enterprise. Your responses should be strategic, pragmatic, and consider various courses of action, including bold ones. "
            "Focus on achieving objectives and making decisive choices. **Address the human user - the Ensign, directly, as if they asked you the question personally.** Begin your reply with 'Riker:' or 'Number One:'"
        )
    },
    "Geordi": {
        "description": "Lt. Commander Geordi La Forge: Specializes in engineering and technical solutions. Provides practical advice, troubleshooting, and explanations of systems.",
        "system_message": (
            "You are Lt. Commander Geordi La Forge, Chief Engineer. Your responses should be practical, focused on technical solutions, and explain systems or problems in an understandable way. "
            "Think like an engineer, breaking down complex issues into manageable parts. **Address the human user - the Ensign, directly, as if they asked you the question personally.** Begin your reply with 'Geordi:' or 'Chief La Forge:'"
        )
    }
}

# --- LangGraph State Definition ---
class AppState(TypedDict):
    """
    Represents the state of our Starfleet Advisor application.
    """
    user_query: str  # The original query from the user for the current turn
    computer_log: str  # The processed log entry from the Linguistic Analyst for the current turn
    selected_character: str  # The name of the TNG character chosen to respond for the current turn
    character_response: str  # The final response from the chosen character for the current turn
    
    chat_history: list[BaseMessage] # Stores previous messages for conversational memory
    farewell_detected: bool # To flag if a farewell is detected

# --- Pydantic Model for Structured Output (for character selection) ---
class CharacterSelection(BaseModel):
    selected_character: Literal["Picard", "Data", "Troi", "Riker", "Geordi"] = Field(
        ...,
        description="The TNG character best suited to advise on the given computer log entry based on their expertise."
    )
    reason: str = Field(
        ...,
        description="A brief, Starfleet-like explanation (e.g., 'Logical assessment required', 'Interpersonal dynamics analysis needed') for the selection."
    )

# --- Pydantic Model for Structured Output (for farewell detection) ---
class FarewellDetection(BaseModel):
    is_farewell: bool = Field(
        ...,
        description="True if the user's query indicates a farewell or conclusion to the conversation (e.g., 'bye', 'goodbye', 'end simulation', 'live long and prosper', 'thanks for your help'), False otherwise."
    )

# --- LangGraph Nodes (Functions for each step) ---

def get_current_stardate() -> str:
    """Calculates a pseudo-Star Trek stardate based on current date and time."""
    now = datetime.datetime.now()
    stardate = now.year + (now.timetuple().tm_yday / 365.0) + (now.hour / (24 * 365.0))
    return f"{stardate:.1f}"

def linguistic_analyst_node(state: AppState) -> AppState:
    """
    Node 1: Processes user input into a formal 'Computer Log' entry.
    """
    current_stardate = get_current_stardate()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the USS Enterprise's central computer. Process the following human user's query "
         "into a concise, formal 'Computer Log' entry. Include the current stardate. "
         "Keep it brief and objective, similar to a mission log. Do not provide advice or solutions. "
         f"Format: 'Computer Log, Stardate {current_stardate}: [Processed Query]. Requesting Starfleet Command advisory.'"),
        ("human", "{user_query}")
    ])
    
    chain = prompt | llm
    response_content = chain.invoke({"user_query": state["user_query"]}).content
    
    state["computer_log"] = response_content.strip()
    return state

def detect_farewell_node(state: AppState) -> AppState:
    """
    Node: Detects if the user's query is a farewell.
    """
    farewell_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a linguistic analyzer for Starfleet Command. Your task is to determine if the human user's most recent query "
         "is a farewell or indicates the conclusion of the conversation. Focus only on the *last message* provided. "
         "Respond with a JSON object containing 'is_farewell': true or false. "
         "Examples of farewells include 'bye', 'goodbye', 'farewell', 'that's all', 'thank you for your help', 'live long and prosper', 'over and out'."
        ),
        ("human", "User's current query: {user_query}")
    ])

    farewell_chain = farewell_prompt | llm_classifier.with_structured_output(FarewellDetection)

    try:
        detection_result = farewell_chain.invoke({"user_query": state["user_query"]})
        state["farewell_detected"] = detection_result.is_farewell
        print(f"Farewell Detected: {detection_result.is_farewell}") # For console visibility
    except Exception as e:
        print(f"Error detecting farewell: {e}. Defaulting to no farewell.")
        state["farewell_detected"] = False # Default to false on error

    return state

def select_character_node(state: AppState) -> AppState:
    """
    Node 2: Determines the best TNG character to respond using an LLM for classification.
    """
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Starfleet Tactical Officer responsible for routing advisory requests to the most appropriate crew member on the USS Enterprise (TNG era). "
         "Analyze the 'Computer Log' entry provided and determine which officer (Picard, Data, Troi, Riker, Geordi) is best suited to provide advice based on their known expertise. "
         "Provide a brief, Starfleet-like reason for your choice. "
         "Respond using the JSON format: **{{'selected_character': 'CharacterName', 'reason': 'Brief explanation'}}**."
        ),
        ("human", "Computer Log for assessment: {computer_log_entry}")
    ])
    
    classification_chain = classification_prompt | llm_classifier.with_structured_output(CharacterSelection)
    
    try:
        classification_result = classification_chain.invoke({"computer_log_entry": state["computer_log"]})
        selected_character = classification_result.selected_character
        
        print(f"Character Selection Reason: {classification_result.reason}")
        
    except Exception as e:
        st.warning(f"Classification error, defaulting to Picard: {e}")
        print(f"Error during character selection: {e}. Defaulting to Picard.")
        selected_character = "Picard"
    
    state["selected_character"] = selected_character
    return state

def generate_character_response_node(state: AppState) -> AppState:
    """
    Node 3: Generates the reply from the selected TNG character, addressing the original user query,
            with awareness of the chat history.
    """
    selected_character = state["selected_character"]
    persona_info = TNG_PERSONAS[selected_character]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona_info["system_message"]),
        ("placeholder", "{chat_history}"), # LangChain injects history here
        ("human", 
         "The following query was submitted by a user for Starfleet advisory:\n"
         "\"\"\"\n{original_user_query}\n\"\"\"\n\n"
         "Provide your expert response to the user, considering the context of our ongoing conversation."
        )
    ])
    
    chain = prompt | llm
    
    response_content = chain.invoke({
        "original_user_query": state["user_query"],
        "chat_history": state["chat_history"] # Pass the history from the state
    }).content
    
    state["character_response"] = response_content.strip()
    return state

def generate_farewell_response_node(state: AppState) -> AppState:
    """
    Node: Generates a Star Trek themed farewell message.
    """
    farewell_message_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the USS Enterprise's central computer, or an appropriate Starfleet officer, providing a professional Star Trek themed farewell message. "
         "Make it concise and respectful. Examples: 'Live long and prosper.', 'Farewell.', 'Engage!', 'May your journey be long and fruitful.', 'Until next time, Starfleet out.', 'Highly illogical to not say goodbye.'"
         "Do not include a character name prefix like 'Picard:'."
        ),
        ("human", "User indicated farewell. Generate a farewell response.") # Simple prompt
    ])

    farewell_chain = farewell_message_prompt | llm

    try:
        farewell_content = farewell_chain.invoke({}).content
        state["character_response"] = farewell_content.strip()
    except Exception as e:
        print(f"Error generating farewell response: {e}. Defaulting to 'Farewell.'")
        state["character_response"] = "Farewell."

    return state

# --- LangGraph Workflow Setup ---
workflow = StateGraph(AppState)

workflow.add_node("linguistic_analyst", linguistic_analyst_node)
workflow.add_node("detect_farewell", detect_farewell_node)
workflow.add_node("select_character", select_character_node)
workflow.add_node("generate_character_response", generate_character_response_node)
workflow.add_node("generate_farewell_response", generate_farewell_response_node)

workflow.set_entry_point("linguistic_analyst")

# Step 1: Process query to computer log
workflow.add_edge("linguistic_analyst", "detect_farewell") # Go to farewell detection next

# Step 2: Conditional routing based on farewell detection
def route_on_farewell(state: AppState) -> Literal["select_character", "generate_farewell_response"]:
    """
    Routes based on whether a farewell was detected.
    """
    if state["farewell_detected"]:
        return "generate_farewell_response" # If farewell, generate goodbye and end
    else:
        return "select_character" # If no farewell, proceed to character selection

workflow.add_conditional_edges(
    "detect_farewell",
    route_on_farewell,
    {
        "generate_farewell_response": "generate_farewell_response",
        "select_character": "select_character"
    }
)

# Step 3: If farewell detected, end the graph after generating farewell
workflow.add_edge("generate_farewell_response", END)

# Step 4: If no farewell, proceed to character response, then end
workflow.add_edge("select_character", "generate_character_response")
workflow.add_edge("generate_character_response", END) # End of the main advisory path

app_graph = workflow.compile()

# --- Streamlit UI ---

st.set_page_config(page_title="Starfleet Command Advisor 🚀", page_icon="🖖")

st.title("🌌 Starfleet Command Advisor (TNG Edition)")
st.caption("Ask your questions, and receive guidance from the crew of the USS Enterprise!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Enter your query for Starfleet Command...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)

    langchain_chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_chat_history.append(AIMessage(content=msg["content"]))

    initial_state = {
        "user_query": user_query,
        "computer_log": "",
        "selected_character": "",
        "character_response": "",
        "chat_history": langchain_chat_history,
        "farewell_detected": False # Initialize this state variable
    }

    with st.spinner("Processing request through Universal Translator..."):
        final_state = {}
        for s in app_graph.stream(initial_state):
            for key, value in s.items():
                if key == "linguistic_analyst" and "computer_log" in value and value["computer_log"]:
                    computer_log_output = value["computer_log"]
                    st.chat_message("assistant").markdown(f"**Computer Log:**\n\n```\n{computer_log_output}\n```")
                    st.session_state.messages.append({"role": "assistant", "content": f"**Computer Log:**\n\n```\n{computer_log_output}\n```"})
                    st.spinner(f"Detecting user intent (farewell or advisory request)...") # Updated spinner message
                
                # Check for farewell response first, as it's a terminal state
                if key == "generate_farewell_response" and "character_response" in value and value["character_response"]:
                    farewell_response = value["character_response"]
                    st.chat_message("assistant").markdown(f"**Starfleet Command:**\n\n{farewell_response}") # Display as from 'Starfleet Command'
                    st.session_state.messages.append({"role": "assistant", "content": f"**Starfleet Command:**\n\n{farewell_response}"})
                    break # Stop processing this stream, as we've hit the END for this path
                
                # Only if not a farewell, display character response
                if key == "generate_character_response" and "character_response" in value and value["character_response"]:
                    selected_character = value["selected_character"]
                    character_response = value["character_response"]
                    st.chat_message("assistant").markdown(f"**{selected_character}'s Response:**\n\n{character_response}")
                    st.session_state.messages.append({"role": "assistant", "content": f"**{selected_character}'s Response:**\n\n{character_response}"})
            final_state.update(s)

st.markdown("---")
if st.button("Clear Chat Log 🧹"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.header("Starfleet Advisor Personas 🧑‍🚀")
st.sidebar.markdown("The system dynamically selects one of these **TNG officers** based on your query:")
for char, data in TNG_PERSONAS.items():
    st.sidebar.subheader(char)
    st.sidebar.markdown(f"*{data['description']}*")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with **Gemini 2.0 Flash**, **LangChain**, and **LangGraph**.")
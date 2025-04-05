import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# Load API key from .env
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset and ensure the Date column is datetime
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"])

# Set the app title
st.title("The Athens Passport 🗺️")

# Initialize conversation history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Helper Functions ---

# Returns events in the given date range, sorted by Date and Time
def get_events_for_date_range(start_date, end_date):
    filtered = events_df[(events_df["Date"] >= start_date) & (events_df["Date"] <= end_date)]
    if filtered.empty:
        return "No events found for this period."
    # Sort events by Date then Time (assuming Time is stored as HH:MM:SS)
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        context += f"• {row['Event']} on {row['Date'].strftime('%A, %B %d, %Y')} at {row['Time']} at {row['Location']}. Price: {row['Price']}\n"
    return context

# For simplicity, we simulate "today" as March 13, 2025 for testing
simulated_today = datetime(2025, 3, 13)
today_str = simulated_today.strftime("%A, %B %d, %Y")

# Calculate this weekend (Saturday and Sunday) based on simulated_today
weekday = simulated_today.weekday()  # Monday=0, ..., Sunday=6
saturday = simulated_today + timedelta(days=(5 - weekday))
sunday = simulated_today + timedelta(days=(6 - weekday))
weekend_str = f"{saturday.strftime('%A, %B %d, %Y')} to {sunday.strftime('%A, %B %d, %Y')}"

# Determine dataset context based on the query.
# For queries mentioning "weekend", we return events for Saturday & Sunday.
def build_dataset_context(query):
    query_lower = query.lower()
    if "weekend" in query_lower or "saturday" in query_lower or "sunday" in query_lower:
        return get_events_for_date_range(saturday, sunday)
    else:
        # Default to events on the simulated "today"
        return get_events_for_date_range(simulated_today, simulated_today)

# --- Main Interaction ---

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    # Build dataset context based on the query
    dataset_context = build_dataset_context(prompt)
    
    # Build conversation history string
    conversation_history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )
    
    # Custom instructions for the assistant:
    custom_instructions = (
        f"Today is {today_str} and we're in the Eastern Time Zone. This weekend is from {weekend_str}. "
        "You're The Athens Passport—a chill, collegiate event and date planning assistant with access to the Athens events dataset. "
        "When answering queries, always use the dataset as your source of truth and reference specific events with their dates, times, and locations exactly as given. "
        "Arrange events in logical, chronological order (earlier events before later ones on the same day). "
        "If a user asks for recommendations (like 'plan a date for me and my girlfriend this weekend'), consider the dataset events first, then add creative suggestions if needed. "
        "Below is the dataset context for the relevant period:\n"
        f"{dataset_context}"
    )
    
    # Build the final prompt that includes conversation history and custom instructions
    full_prompt = (
        f"{custom_instructions}\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"User: {prompt}\n"
        f"Assistant (in a chill tone):"
    )
    
    # Save the user's query in the session history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get the response from the LLM using our full prompt
    with st.chat_message("assistant"):
        response = chat_engine.chat(full_prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


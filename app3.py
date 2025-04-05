import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# Load API key from .env
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; you can switch to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset and ensure the Date column is parsed as datetime
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

def format_price(price):
    try:
        price_value = float(price)
        if price_value == 0:
            return "Free"
        else:
            return f"${price_value:.2f}"
    except Exception:
        return str(price)

def format_time(time_str):
    try:
        # Convert "HH:MM:SS" to a datetime object then to 12-hour format (e.g., "8:00 AM")
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        return time_obj.strftime("%-I:%M %p")
    except Exception:
        return time_str

def get_events_for_date_range(start_date, end_date):
    filtered = events_df[(events_df["Date"] >= start_date) & (events_df["Date"] <= end_date)]
    if filtered.empty:
        return "No events found for this period."
    # Sort events by Date then Time
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']}. Price: {price_str}\n"
    return context

# For testing, simulate "today" as March 13, 2025
simulated_today = datetime(2025, 3, 13)
today_str = simulated_today.strftime("%A, %B %d, %Y")

# Define this weekend based on simulated_today:
weekday = simulated_today.weekday()  # Monday=0, ..., Sunday=6
saturday = simulated_today + timedelta(days=(5 - weekday))
sunday = simulated_today + timedelta(days=(6 - weekday))
weekend_str = f"{saturday.strftime('%A, %B %d, %Y')} to {sunday.strftime('%A, %B %d, %Y')}"

# Build dataset context based on the query
def build_dataset_context(query):
    query_lower = query.lower()
    # If the query mentions weekend, Saturday, or Sunday, use weekend context.
    if "weekend" in query_lower or "saturday" in query_lower or "sunday" in query_lower:
        return get_events_for_date_range(saturday, sunday)
    else:
        # Otherwise, default to events on simulated_today
        return get_events_for_date_range(simulated_today, simulated_today)

# --- Main Interaction ---

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    # Get dataset context based on the query
    dataset_context = build_dataset_context(prompt)
    
    # Build conversation history string
    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    
    # Custom instructions for the assistant:
    custom_instructions = (
        f"Hey, it's {today_str} and we're in the Eastern Time Zone. This weekend is from {weekend_str}. "
        "You're The Athens Passport—a chill, collegiate event and date planning assistant with access to the Athens events dataset. "
        "When answering queries, please use the dataset as your source of truth, referencing specific events with details like event name, location, "
        "and time (using standard times, e.g., '8:00 AM' not military time). Arrange events in logical, chronological order. "
        "When a user asks for recommendations (like 'plan a date for me and my girlfriend this weekend'), "
        "consider the dataset events first and blend them with creative suggestions. "
        "Below is the dataset context for the relevant period:\n"
        f"{dataset_context}"
    )
    
    # Build the final prompt
    full_prompt = (
        f"{custom_instructions}\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"User: {prompt}\n"
        f"Assistant (in a chill tone):"
    )
    
    # Save the user's query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = chat_engine.chat(full_prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


import streamlit as st
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# Load API key from .env
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; change to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset from Excel and ensure Date is parsed correctly
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"])

# --- Helper Functions ---

def format_price(price):
    try:
        price_value = float(price)
        return "Free" if price_value == 0 else f"${price_value:.2f}"
    except Exception:
        return str(price)

def format_time(time_str):
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        return time_obj.strftime("%-I:%M %p")
    except Exception:
        return time_str

def get_events_for_date_range(start_date, end_date):
    filtered = events_df[(events_df["Date"] >= start_date) & (events_df["Date"] <= end_date)]
    if filtered.empty:
        return "No events found for this period."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']}. Price: {price_str}\n"
    return context

def get_events_for_category(category):
    filtered = events_df[events_df["Category"].str.lower() == category.lower()]
    if filtered.empty:
        return f"No events found for {category}."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']}. Price: {price_str}\n"
    return context

def build_dataset_context(query):
    query_lower = query.lower()
    # If query asks about karaoke, filter by category
    if "karaoke" in query_lower:
        return get_events_for_category("Karaoke & Open Mic")
    # If query mentions weekend (or Saturday/Sunday), use weekend events
    elif "weekend" in query_lower or "saturday" in query_lower or "sunday" in query_lower:
        return get_events_for_date_range(saturday, sunday)
    else:
        # Default: return events for the simulated "today"
        return get_events_for_date_range(simulated_today, simulated_today)

# --- Context Setup ---

# For testing, simulate "today" as March 13, 2025.
simulated_today = datetime(2025, 3, 13)
today_str = simulated_today.strftime("%A, %B %d, %Y")

# Calculate this weekend (assumed: Saturday and Sunday)
weekday = simulated_today.weekday()  # Monday=0,..., Sunday=6
saturday = simulated_today + timedelta(days=(5 - weekday))
sunday = simulated_today + timedelta(days=(6 - weekday))
weekend_str = f"{saturday.strftime('%A, %B %d, %Y')} to {sunday.strftime('%A, %B %d, %Y')}"

# --- Main Interaction ---

st.title("The Athens Passport 🗺️")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    # Build dataset context based on query
    dataset_context = build_dataset_context(prompt)
    
    conversation_history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )
    
    custom_instructions = (
        f"Hey, it's {today_str} and we're in the Eastern Time Zone. This weekend is from {weekend_str}. "
        "You're The Athens Passport—a chill, collegiate event and date planning assistant with access to the Athens events dataset. "
        "When answering queries, use the dataset as your source of truth and arrange events in logical, chronological order (using standard times, e.g., '8:00 AM'). "
        "When a user asks for recommendations—like 'plan a date for me and my girlfriend this weekend'—you should consider the dataset events first and blend them with creative suggestions. "
        "Below is the dataset context for the relevant period:\n"
        f"{dataset_context}"
    )
    
    full_prompt = (
        f"{custom_instructions}\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"User: {prompt}\n"
        f"Assistant (in a chill tone):"
    )
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = chat_engine.chat(full_prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


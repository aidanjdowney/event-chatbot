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

# Set up the OpenAI LLM (using GPT-3.5-turbo; switch to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset from Excel and ensure the Date column is parsed as datetime
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"])

# Set the app title
st.title("The Athens Passport 🗺️")

# Initialize conversation history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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

def get_events_for_date_range(target_date):
    # Filter events for target_date by comparing only the date portion
    filtered = events_df[events_df["Date"].dt.date == target_date.date()]
    if filtered.empty:
        return "No events found for this date."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']}. Price: {price_str}\n"
    return context

def get_events_for_category_and_date(category, target_date):
    filtered = events_df[(events_df["Category"].str.lower() == category.lower()) &
                         (events_df["Date"].dt.date == target_date.date())]
    if filtered.empty:
        return f"No {category} events found on {target_date.strftime('%A, %B %d, %Y')}."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']}. Price: {price_str}\n"
    return context

def determine_target_date(query):
    query_lower = query.lower()
    # If the query mentions "next week", return next Monday (for a week-long range)
    if "next week" in query_lower:
        next_monday = simulated_today + timedelta(days=(7 - simulated_today.weekday()))
        return next_monday
    # If query explicitly mentions "saturday" or "sunday", choose that day
    if "saturday" in query_lower:
        return simulated_today + timedelta(days=(5 - simulated_today.weekday()))
    elif "sunday" in query_lower:
        return simulated_today + timedelta(days=(6 - simulated_today.weekday()))
    # Look for an explicit date in MM/DD/YY or MM/DD/YYYY format
    match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", query)
    if match:
        date_str = match.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    # If query mentions "weekend" without a specific day, default to Saturday
    if "weekend" in query_lower:
        return simulated_today + timedelta(days=(5 - simulated_today.weekday()))
    # Default to simulated_today
    return simulated_today

def build_dataset_context(query, target_date):
    query_lower = query.lower()
    if "karaoke" in query_lower:
        return get_events_for_category_and_date("Karaoke & Open Mic", target_date)
    elif "music" in query_lower or "concert" in query_lower:
        return get_events_for_category_and_date("Music", target_date)
    elif "comedy" in query_lower:
        return get_events_for_category_and_date("Comedy", target_date)
    else:
        return get_events_for_date_range(target_date)

# --- Context Setup ---

# For testing, simulate "today" as March 13, 2025.
simulated_today = datetime(2025, 3, 13)
today_str = simulated_today.strftime("%A, %B %d, %Y")

# Define this weekend (Saturday and Sunday) based on simulated_today
weekday = simulated_today.weekday()  # Monday=0, ..., Sunday=6
this_saturday = simulated_today + timedelta(days=(5 - weekday))
this_sunday = simulated_today + timedelta(days=(6 - weekday))
weekend_str = f"{this_saturday.strftime('%A, %B %d, %Y')} to {this_sunday.strftime('%A, %B %d, %Y')}"

# --- Main Interaction ---

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    target_date = determine_target_date(prompt)
    dataset_context = build_dataset_context(prompt, target_date)
    
    # Choose context wording based on query specifics
    query_lower = prompt.lower()
    if "next week" in query_lower:
        next_monday = simulated_today + timedelta(days=(7 - simulated_today.weekday()))
        next_sunday = next_monday + timedelta(days=6)
        date_context_text = f"for next week (Monday: {next_monday.strftime('%A, %B %d, %Y')} to Sunday: {next_sunday.strftime('%A, %B %d, %Y')})"
    elif "saturday" in query_lower or "sunday" in query_lower:
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    elif "weekend" in query_lower:
        date_context_text = f"for the weekend (Saturday: {this_saturday.strftime('%A, %B %d, %Y')}, Sunday: {this_sunday.strftime('%A, %B %d, %Y')})"
    else:
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    
    conversation_history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )
    
    custom_instructions = (
        f"Hey, it's {today_str} and we're in the Eastern Time Zone. {date_context_text}. "
        "You're The Athens Passport—a chill, collegiate event and date planning assistant with access to the Athens events dataset. "
        "When answering queries, please base your answer on the dataset as much as possible and arrange events in strict chronological order "
        "(morning events first, then afternoon, then evening; use standard times like '8:00 AM'). "
        "When a user asks for recommendations—like 'plan a date for me and my girlfriend this weekend'—select events from the dataset in logical order "
        "and blend them with creative suggestions. "
        "Below is the dataset context for the specified period:\n"
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


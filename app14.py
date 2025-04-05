import os
# Set the API key explicitly from Streamlit secrets
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
import streamlit as st
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# Load API key from .env file
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; change to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset from Excel and parse the Date column
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"])

# Set the app title to "The Guide Dawg"
st.title("The Guide Dawg")

# Initialize conversation history and last_target_date in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_target_date" not in st.session_state:
    st.session_state.last_target_date = None

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
    # Filter events that occur exactly on target_date (comparing only the date portion)
    filtered = events_df[events_df["Date"].dt.date == target_date.date()]
    if filtered.empty:
        return "No events found for this date."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
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
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

def get_category_events_for_date_range(category, start_date, end_date):
    filtered = events_df[(events_df["Category"].str.lower() == category.lower()) &
                         (events_df["Date"].dt.date >= start_date.date()) &
                         (events_df["Date"].dt.date <= end_date.date())]
    if filtered.empty:
        return f"No {category} events found from {start_date.strftime('%A, %B %d, %Y')} to {end_date.strftime('%A, %B %d, %Y')}."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

def get_events_for_date_range_range(start_date, end_date):
    filtered = events_df[(events_df["Date"].dt.date >= start_date.date()) &
                         (events_df["Date"].dt.date <= end_date.date())]
    if filtered.empty:
        return "No events found for this period."
    filtered = filtered.sort_values(by=["Date", "Time"])
    context = ""
    for _, row in filtered.iterrows():
        event_date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str_formatted = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        context += f"• {row['Event']} on {event_date_str} at {time_str_formatted} at {row['Location']} — {price_str}\n"
    return context

def determine_target_date(query, base_date):
    query_lower = query.lower()
    # If the query includes "that day", "later", "other", or "that night", reuse the stored target date.
    if st.session_state.get("last_target_date") and ("that day" in query_lower or "later" in query_lower or "other" in query_lower or "that night" in query_lower):
        return st.session_state["last_target_date"]
    # If query mentions "next week", return next Monday
    if "next week" in query_lower:
        next_monday = base_date + timedelta(days=(7 - base_date.weekday()))
        return next_monday
    # Check for specific day names (e.g., "wednesday")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if day in query_lower:
            today_index = base_date.weekday()
            target_index = days.index(day)
            days_ahead = target_index - today_index if target_index >= today_index else target_index - today_index + 7
            return base_date + timedelta(days=days_ahead)
    # Look for an explicit date in MM/DD/YY or MM/DD/YYYY
    match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", query)
    if match:
        date_str = match.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    # If query mentions "weekend", default to Saturday
    if "weekend" in query_lower:
        return base_date + timedelta(days=(5 - base_date.weekday()))
    # Otherwise, default to base_date
    return base_date

def build_dataset_context(query, target_date):
    query_lower = query.lower()
    # For range queries like "next week", build context for the entire week
    if "next week" in query_lower:
        next_sunday = target_date + timedelta(days=6)
        if "karaoke" in query_lower:
            return get_category_events_for_date_range("Karaoke & Open Mic", target_date, next_sunday)
        elif "music" in query_lower or "concert" in query_lower:
            return get_category_events_for_date_range("Music", target_date, next_sunday)
        elif "comedy" in query_lower:
            return get_category_events_for_date_range("Comedy", target_date, next_sunday)
        else:
            return get_events_for_date_range_range(target_date, next_sunday)
    # Otherwise, if a specific category is mentioned, filter for that day.
    if "karaoke" in query_lower:
        return get_events_for_category_and_date("Karaoke & Open Mic", target_date)
    elif "music" in query_lower or "concert" in query_lower:
        return get_events_for_category_and_date("Music", target_date)
    elif "comedy" in query_lower:
        return get_events_for_category_and_date("Comedy", target_date)
    else:
        return get_events_for_date_range(target_date)

# --- Main Interaction ---

current_date = datetime.today()
today_str = current_date.strftime("%A, %B %d, %Y")

weekday = current_date.weekday()
this_saturday = current_date + timedelta(days=(5 - weekday))
this_sunday = current_date + timedelta(days=(6 - weekday))
weekend_str = f"{this_saturday.strftime('%A, %B %d, %Y')} to {this_sunday.strftime('%A, %B %d, %Y')}"

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    target_date = determine_target_date(prompt, current_date)
    st.session_state["last_target_date"] = target_date
    dataset_context = build_dataset_context(prompt, target_date)
    
    query_lower = prompt.lower()
    if "next week" in query_lower:
        next_monday = current_date + timedelta(days=(7 - current_date.weekday()))
        next_sunday = next_monday + timedelta(days=6)
        date_context_text = f"for next week (Monday: {next_monday.strftime('%A, %B %d, %Y')} to Sunday: {next_sunday.strftime('%A, %B %d, %Y')})"
    elif any(term in query_lower for term in ["saturday", "sunday", "that day", "later", "other", "that night"]):
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    elif "weekend" in query_lower:
        date_context_text = f"for the weekend (Saturday: {this_saturday.strftime('%A, %B %d, %Y')}, Sunday: {this_sunday.strftime('%A, %B %d, %Y')})"
    else:
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    
    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    
    custom_instructions = (
        f"Hey, it's {today_str} and we're in the Eastern Time Zone. {date_context_text}. "
        "You're The Guide Dawg—your Athens events guru with access to the Athens events dataset. "
        "When answering queries, please base your answer solely on the dataset provided below and arrange events in strict chronological order "
        "(morning events first, then afternoon, then evening; use standard times like '8:00 AM'). "
        "For date planning, select a curated itinerary (one or two events per time block) and blend them with creative suggestions. "
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


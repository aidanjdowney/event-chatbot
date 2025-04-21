import streamlit as st
import os
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI

# -------------------------------------------------------------------
# 1. SETUP AND INITIALIZATION
# -------------------------------------------------------------------

# Set the API key explicitly from Streamlit secrets (ensure your secrets.toml includes your key)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Load environment variables (optional, for local development)
load_dotenv()

# Set up the OpenAI LLM (using GPT-4 in this version)
Settings.llm = OpenAI(model="gpt-4")

# Load the stored index built from your Athens events dataset
storage_context = StorageContext.from_defaults(persist_dir="./athens_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load the events dataset from Excel and convert "Date" to a plain date
events_df = pd.read_excel("athens_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"], errors="coerce").dt.date

# Set the app title
st.title("The Guide Dawg 🐾")

# Initialize session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_target_date" not in st.session_state:
    st.session_state.last_target_date = None

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------------

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

def format_events_simple_list(df: pd.DataFrame) -> str:
    """Return a bullet list of events in chronological order."""
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["Date", "Time"])
    lines = []
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%A, %B %d, %Y")
        time_str = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        line = f"- {row['Event']} on {date_str} at {time_str} @ {row['Location']} ({price_str})"
        lines.append(line)
    return "\n".join(lines)

def group_events_by_day(df: pd.DataFrame) -> str:
    """Group events by Date with bold headings per day."""
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["Date", "Time"])
    grouped_text = ""
    current_day = None
    for _, row in df.iterrows():
        day_date = row["Date"]
        if day_date != current_day:
            if grouped_text:
                grouped_text += "\n\n"
            day_str = day_date.strftime("%A, %B %d, %Y")
            grouped_text += f"**{day_str}**\n"
            current_day = day_date
        time_str = format_time(str(row["Time"]))
        price_str = format_price(row["Price"])
        bullet_line = f"- {row['Event']} at {time_str} @ {row['Location']} ({price_str})"
        grouped_text += bullet_line + "\n"
    return grouped_text.strip()

def filter_events(category=None, start_date=None, end_date=None, location_substring=None) -> pd.DataFrame:
    """Return a DataFrame of events filtered by category, date range, etc."""
    df = events_df.copy()
    if category:
        df = df[df["Category"].fillna("").str.lower() == category.lower()]
    if location_substring:
        df = df[df["Location"].fillna("").str.lower().str.contains(location_substring.lower())]
    if start_date and end_date:
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    elif start_date:
        df = df[df["Date"] == start_date]
    return df

def get_next_week_range():
    """Return next Monday through next Sunday."""
    today = datetime.today().date()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)
    next_sunday = next_monday + timedelta(days=6)
    return next_monday, next_sunday

def get_next_weekend():
    """Return the upcoming Saturday and Sunday."""
    today = datetime.today().date()
    days_until_saturday = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)
    return saturday, sunday

def get_grouped_events_for_date_range_range(start_date, end_date):
    current = start_date
    all_context = ""
    while current <= end_date:
        events = get_events_for_date_range(current)
        day_str = current.strftime("%A, %B %d, %Y")
        events_text = events if not events.startswith("No events") else "No events."
        all_context += f"{day_str}:\n{events_text}\n"
        current += timedelta(days=1)
    return all_context

def get_grouped_category_events_for_date_range(category, start_date, end_date):
    current = start_date
    all_context = ""
    while current <= end_date:
        events = get_events_for_category_and_date(category, current)
        day_str = current.strftime("%A, %B %d, %Y")
        events_text = events if not events.startswith(f"No {category}") else "No events."
        all_context += f"{day_str}:\n{events_text}\n"
        current += timedelta(days=1)
    return all_context

def get_events_for_date_range(target_date):
    filtered = events_df[events_df["Date"] == target_date]
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
                         (events_df["Date"] == target_date)]
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

def get_events_for_date_range_range(start_date, end_date):
    filtered = events_df[(events_df["Date"] >= start_date) & (events_df["Date"] <= end_date)]
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
    if "tomorrow" in query_lower:
        return base_date + timedelta(days=1)
    if st.session_state.get("last_target_date") and any(term in query_lower for term in ["that day", "later", "other", "that night"]):
        return st.session_state["last_target_date"]
    if "next week" in query_lower:
        next_monday = base_date + timedelta(days=(7 - base_date.weekday()))
        return next_monday
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if day in query_lower:
            today_index = base_date.weekday()
            target_index = days.index(day)
            days_ahead = target_index - today_index if target_index >= today_index else target_index - today_index + 7
            return base_date + timedelta(days=days_ahead)
    match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", query)
    if match:
        date_str = match.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
    if "weekend" in query_lower:
        return base_date + timedelta(days=(5 - base_date.weekday()))
    return base_date

def build_dataset_context(query, target_date):
    query_lower = query.lower()
    if "next week" in query_lower:
        next_sunday = target_date + timedelta(days=6)
        if "karaoke" in query_lower:
            return get_grouped_category_events_for_date_range("Karaoke & Open Mic", target_date, next_sunday)
        elif "music" in query_lower or "concert" in query_lower:
            return get_grouped_category_events_for_date_range("Music", target_date, next_sunday)
        elif "comedy" in query_lower:
            return get_grouped_category_events_for_date_range("Comedy", target_date, next_sunday)
        else:
            return get_grouped_events_for_date_range_range(target_date, next_sunday)
    elif "weekend" in query_lower:
        # Explicitly build context for both Saturday and Sunday.
        saturday, sunday = get_next_weekend()
        saturday_events = get_events_for_date_range(saturday)
        sunday_events = get_events_for_date_range(sunday)
        context = (
            f"{saturday.strftime('%A, %B %d, %Y')}:\n{saturday_events}\n\n"
            f"{sunday.strftime('%A, %B %d, %Y')}:\n{sunday_events}"
        )
        return context
    elif "karaoke" in query_lower:
        df = filter_events(category="Karaoke & Open Mic", start_date=target_date, end_date=target_date)
        return format_events_simple_list(df)
    elif "music" in query_lower or "concert" in query_lower:
        df = filter_events(category="Music", start_date=target_date, end_date=target_date)
        return format_events_simple_list(df)
    elif "comedy" in query_lower:
        df = filter_events(category="Comedy", start_date=target_date, end_date=target_date)
        return format_events_simple_list(df)
    else:
        df = filter_events(start_date=target_date, end_date=target_date)
        return format_events_simple_list(df)

# -------------------------------------------------------------------
# 3. MAIN APPLICATION LOGIC
# -------------------------------------------------------------------

current_date = datetime.today().date()
today_str = current_date.strftime("%A, %B %d, %Y")

# Pre-calculate weekend string using upcoming Saturday & Sunday
weekday = current_date.weekday()
this_saturday = current_date + timedelta(days=(5 - weekday) % 7)
this_sunday = this_saturday + timedelta(days=1)
weekend_str = f"{this_saturday.strftime('%A, %B %d, %Y')} to {this_sunday.strftime('%A, %B %d, %Y')}"

if prompt := st.chat_input("Ask me about Athens events or plan a date:"):
    prompt_lower = prompt.lower()
    
    # Intercept queries about "who made you" or "who created you"
    if "who made you" in prompt_lower or "who created you" in prompt_lower:
        direct_response = "I was created by three MSBA students at UGA: Sam Toole, Aidan Downey, and Jacob Croskey."
        with st.chat_message("assistant"):
            st.markdown(direct_response)
            st.session_state.messages.append({"role": "assistant", "content": direct_response})
        st.stop()
    
    # Determine if the query is asking for a date plan
    wants_date_plan = ("plan a date" in prompt_lower or "date night" in prompt_lower)
    target_date = determine_target_date(prompt, current_date)
    st.session_state["last_target_date"] = target_date
    dataset_context = build_dataset_context(prompt, target_date)
    
    # Set context string for the date period
    if "next week" in prompt_lower:
        next_monday, next_sunday = get_next_week_range()
        date_context_text = f"for next week (Monday: {next_monday.strftime('%A, %B %d, %Y')} to Sunday: {next_sunday.strftime('%A, %B %d, %Y')})"
    elif "weekend" in prompt_lower:
        date_context_text = f"for the weekend (Saturday: {this_saturday.strftime('%A, %B %d, %Y')}, Sunday: {this_sunday.strftime('%A, %B %d, %Y')})"
    else:
        date_context_text = f"for {target_date.strftime('%A, %B %d, %Y')}"
    
    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    
    # ----------------------------------------------------------------
    # CUSTOM INSTRUCTIONS FOR THE LLM
    # ----------------------------------------------------------------
    extra_date_instructions = ""
    if wants_date_plan:
        extra_date_instructions = (
            "The user wants a creative date plan. Propose an original itinerary that mixes a few events "
            "from the dataset with your own local recommendations (e.g., dining or cultural spots). "
            "Avoid rigid time blocks and ensure events are spaced out realistically. Be imaginative and unique."
        )
    
    custom_instructions = (
        f"Hey, it's {today_str} in the Eastern Time Zone, {date_context_text}. "
        "You're The Guide Dawg 🐾—a chill, collegiate event and date planning assistant with access to the Athens events dataset. "
        "When someone asks 'What are you?', you may respond with a friendly greeting and mention you're The Guide Dawg. "
        "If asked 'What is your purpose?', say: 'My purpose is to help UGA students and the broader Athens community easily discover local events, "
        "enriching the campus experience and fostering a vibrant, connected community.' "
        "For purely informational queries, simply list the events in chronological order. "
        "If a query refers to 'this weekend', show events for Saturday & Sunday. "
        "If a query mentions a specific location, list those events. "
        "For 'next week' queries, group events by day in chronological order. "
        "If asked to plan a date, propose a creative itinerary using some events from the dataset and supplement with your own recommendations. "
        "Avoid always starting with the same template—be imaginative and original. "
        f"{extra_date_instructions}\n\n"
        "Below is the relevant dataset context:\n"
        f"{dataset_context}"
    )
    
    final_query = (
        f"{custom_instructions}\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"User: {prompt}\n"
        "Assistant (in a chill tone):"
    )
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = chat_engine.chat(final_query)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

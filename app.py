import streamlit as st
import os
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv
import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import pytz
tz = pytz.timezone('America/New_York')

# Set the API key explicitly from Streamlit secrets (make sure your secrets.toml includes your key)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Load environment variables from .env (optional, for local development) 
load_dotenv()

# Set up the OpenAI LLM (using GPT-3.5-turbo; change to GPT-4 if desired)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# Load the stored index and create chat engine
storage_context = StorageContext.from_defaults(persist_dir="./winterville_events_index")
index = load_index_from_storage(storage_context)
chat_engine = index.as_chat_engine(chat_mode="context")

# Load and parse events data
events_df = pd.read_excel("winterville_events.xlsx")
events_df["Date"] = pd.to_datetime(events_df["Date"], errors="coerce").dt.date

st.title("The Winterville Guide 🎨")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_target_date" not in st.session_state:
    st.session_state.last_target_date = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def format_time_str(time_val):
    if pd.isnull(time_val):
        return ""
    try:
        return datetime.strptime(str(time_val), "%H:%M:%S").strftime("%-I:%M %p")
    except:
        return str(time_val)

def format_events_simple_list(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No events found._"
    df = df.sort_values(["Date","Time"])
    lines = []
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%A, %B %d, %Y") if row["Date"] else ""
        time_str = format_time_str(row["Time"])
        price_val = row.get("Price", 0)
        if pd.notnull(price_val):
            try:
                pval = float(price_val)
                price_str = "Free" if pval == 0 else f"${pval:.2f}"
            except:
                price_str = str(price_val)
        else:
            price_str = "Free"
        line = f"- {row['Event']} on {date_str} at {time_str} @ {row['Location']} ({price_str})"
        lines.append(line)
    return "\n".join(lines)

def group_events_by_day(df: pd.DataFrame) -> str:
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
        time_str = format_time_str(row["Time"])
        price_val = row.get("Price", 0)
        if pd.notnull(price_val):
            try:
                pval = float(price_val)
                price_str = "Free" if pval == 0 else f"${pval:.2f}"
            except:
                price_str = str(price_val)
        else:
            price_str = "Free"
        bullet_line = f"- {row['Event']} at {time_str} @ {row['Location']} ({price_str})"
        grouped_text += bullet_line + "\n"
    return grouped_text.strip()

def filter_events(category=None, start_date=None, end_date=None, location_substring=None) -> pd.DataFrame:
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
    today = datetime.now(tz).date()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)
    next_sunday = next_monday + timedelta(days=6)
    return next_monday, next_sunday

def get_next_weekend():
    today = datetime.now(tz).date()
    days_until_saturday = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_until_saturday)
    sunday = saturday + timedelta(days=1)
    return saturday, sunday

def parse_day_of_week(prompt_text: str):
    prompt_lower = prompt_text.lower()
    today = datetime.now(tz).date()
    if "tomorrow" in prompt_lower:
        return today + timedelta(days=1)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in prompt_lower:
            day_diff = (i - today.weekday()) % 7
            if day_diff == 0:
                day_diff = 7
            return today + timedelta(days=day_diff)
    return None

if prompt := st.chat_input("Ask me about Winterville events or plan a date..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_lower = prompt.lower()
    if "who made you" in prompt_lower or "who created you" in prompt_lower:
        direct_response = "I was created by three MSBA students at UGA: Sam Toole, Aidan Downey, and Jacob Croskey."
        with st.chat_message("assistant"):
            st.markdown(direct_response)
            st.session_state.messages.append({"role": "assistant", "content": direct_response})
        st.stop()

    wants_date_plan = ("plan a date" in prompt_lower or "date night" in prompt_lower)

    if "what is today" in prompt_lower:
        today_str = datetime.now(tz).strftime("%A, %B %d, %Y")
        response_text = f"Today is {today_str}!"
        with st.chat_message("assistant"):
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content":response_text})
        st.stop()

    elif "next week" in prompt_lower:
        category = None
        if "music" in prompt_lower:
            category = "Music"
        elif "comedy" in prompt_lower:
            category = "Comedy"
        elif "karaoke" in prompt_lower:
            category = "Karaoke & Open Mic"

        next_monday, next_sunday = get_next_week_range()
        df_next_week = filter_events(category=category, start_date=next_monday, end_date=next_sunday)
        events_text = group_events_by_day(df_next_week)
        event_lines = events_text.splitlines()
        trimmed_events_text = "\n".join(event_lines[:40])
        dataset_context = f"Events for next week (Monday {next_monday} -> Sunday {next_sunday}):\n\n{trimmed_events_text}"

    elif "weekend" in prompt_lower:
        category = None
        if "music" in prompt_lower:
            category = "Music"
        elif "comedy" in prompt_lower:
            category = "Comedy"
        elif "karaoke" in prompt_lower:
            category = "Karaoke & Open Mic"

        sat, sun = get_next_weekend()
        df_weekend = filter_events(category=category, start_date=sat, end_date=sun)
        events_text = group_events_by_day(df_weekend)
        event_lines = events_text.splitlines()
        trimmed_events_text = "\n".join(event_lines[:40])
        dataset_context = f"This weekend (Saturday {sat} & Sunday {sun}):\n\n{trimmed_events_text}"

    else:
        category = None
        if "music" in prompt_lower:
            category = "Music"
        elif "comedy" in prompt_lower:
            category = "Comedy"
        elif "karaoke" in prompt_lower:
            category = "Karaoke & Open Mic"

        location_substring = None
        location_match = re.search(r'events.*?(?:at|in)\s+([A-Za-z0-9\&\-\']+.*)', prompt_lower)
        if location_match:
            location_substring = location_match.group(1).strip()

        if wants_date_plan:
            day_date = parse_day_of_week(prompt)
            if not day_date:
                day_date = datetime.today().date() + timedelta(days=1)
            df_day = filter_events(category=category, start_date=day_date, end_date=day_date, location_substring=location_substring)
            events_text = format_events_simple_list(df_day)
            trimmed_events_text = "\n".join(events_text.splitlines()[:40])
            date_str = day_date.strftime("%A, %B %d, %Y")
            dataset_context = (
                f"You want a creative date plan for {date_str}.\n\n"
                f"Below is a list of events happening that day:\n\n{trimmed_events_text}"
            )
        else:
            df_upcoming = filter_events(category=category, location_substring=location_substring)
            events_text = format_events_simple_list(df_upcoming)
            trimmed_events_text = "\n".join(events_text.splitlines()[:40])
            dataset_context = f"Here are the upcoming events:\n\n{trimmed_events_text}"

    today_str = datetime.now(tz).strftime("%A, %B %d, %Y")
    date_context_text = "Based on your query"

    extra_date_instructions = ""
    if wants_date_plan:
        extra_date_instructions = (
            "The user wants a creative date plan. Avoid labeling it as 'morning/afternoon/evening'—"
            "just choose a few interesting events from the dataset and propose a unique itinerary. "
            "Feel free to add a dinner suggestion from your internal knowledge. "
            "Add fun transitions or commentary—be imaginative!"
        )

    custom_instructions = (
        f"Hey, it's {today_str} and we're in the Eastern Time Zone. {date_context_text}. "
        "You're The Winterville Guide 🎨—a chill event and date planning assistant with access to the Winterville events dataset. "
        "When someone asks 'What are you?', you may respond with a friendly greeting and mention that you're The Winterville Guide. "
        "If asked 'What is your purpose?', say: 'My purpose is to help UGA students and the broader Winterville community easily discover local events.' "
        "If asked 'Who made you?' or 'Who created you?', your code is already intercepting that for a direct short answer. "
        "For purely informational queries (like 'What events are happening X day?'), list them in chronological order. "
        "If a query refers to 'this weekend', show events for Saturday & Sunday. "
        "If a query mentions a specific location, list those events. "
        "For 'next week' queries, group events by day. "
        "If asked to plan a date, propose a creative itinerary using a few events from the dataset. "
        "Don't do a strict 'morning/afternoon/evening' formula. "
        "You can recommend dinner spots from your knowledge of Winterville. "
        "Ensure your final output is well-organized, consistent, and uses plain text. "
        f"{extra_date_instructions}\n\n"
        "Below is the relevant dataset context:\n"
        f"{dataset_context}"
    )

    final_query = (
        f"{custom_instructions}\n\n"
        f"User's prompt: {prompt}\n\n"
        "Assistant:"
    )

    llm_response = chat_engine.chat(final_query)

    with st.chat_message("assistant"):
        st.markdown(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

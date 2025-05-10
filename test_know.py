import streamlit as st
from litellm import completion
import json
import spacy
from rapidfuzz import fuzz

# --- Load NLP model ---
nlp = spacy.load("en_core_web_sm")

# --- Load Hamid's knowledge base ---
with open("hamid_knowledge_base.json", "r") as f:
    hamid_data = json.load(f)

def flatten_json(data, parent_key=''):
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            items.extend(flatten_json(v, new_key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_json(v, new_key))
    else:
        items.append((parent_key, str(data)))
    return items

flat_kb = flatten_json(hamid_data)

def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.is_alpha and not token.is_stop]

def fuzzy_search(keywords, threshold=70):
    results = []
    for key, value in flat_kb:
        for keyword in keywords:
            score = fuzz.partial_ratio(keyword.lower(), value.lower())
            if score >= threshold:
                results.append((score, key, value))
    results.sort(reverse=True)
    return results

def get_hamid_response(user_input):
    input_lower = user_input.lower()

    if "who is hamid" in input_lower:
        return hamid_data["bio"]
    elif "what does hamid do" in input_lower:
        return f"Hamid is a {hamid_data['profession']} skilled in {', '.join(hamid_data['skills'])}."
    elif "contact" in input_lower:
        return f"Email: {hamid_data['contact']['email']}, Website: {hamid_data['contact']['website']}"

    keywords = extract_keywords(user_input)
    matches = fuzzy_search(keywords)

    if matches:
        top_matches = [f"{key}: {value}" for _, key, value in matches[:3]]
        return "Here's what I found:\n" + "\n".join(top_matches)
    
    return None  # Let Gemini handle the rest

# --- Streamlit Setup ---
GOOGLE_API_KEY = "AIzaSyAIN0ZmIiSkRkb0h2y0PIp3YBM0p_N9lV8"
st.set_page_config(page_title="Gemini Chatbot", layout="centered")
st.title("🤖 Hamid's Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Say something...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # --- Try custom response first ---
    custom_response = get_hamid_response(user_prompt)

    if custom_response:
        bot_reply = custom_response
    else:
        try:
            response = completion(
                model="gemini/gemini-1.5-flash",
                messages=st.session_state.messages,
                api_key=GOOGLE_API_KEY
            )
            bot_reply = response['choices'][0]['message']['content']
        except Exception as e:
            bot_reply = f"❌ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

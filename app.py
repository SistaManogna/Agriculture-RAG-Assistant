"""A simple, source-backed agriculture chat interface."""

import logging
import base64
import html
import re
import hashlib
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Agri • Agriculture Assistant", page_icon="🌱", layout="wide")

DOCUMENT_NAMES = {
    "01_soil_health_management.pdf": "Soil Health Management",
    "02_npk_fertilizer_management.pdf": "NPK Fertilizer Management",
    "03_irrigation_water_management.pdf": "Irrigation & Water Management",
    "04_integrated_pest_management.pdf": "Integrated Pest Management",
    "05_crop_rotation_and_diversification.pdf": "Crop Rotation & Diversification",
    "06_precision_agriculture.pdf": "Precision Agriculture",
    "07_climate_smart_agriculture.pdf": "Climate-Smart Agriculture",
    "08_organic_and_sustainable_farming.pdf": "Organic & Sustainable Farming",
}
STARTERS = {
    "🌱 Soil health": "How can farmers improve soil health?",
    "💧 Irrigation": "How can farmers improve water-use efficiency?",
    "🌾 Crop rotation": "What are the benefits of crop rotation?",
    "📡 Smart farming": "How does precision agriculture help farmers?",
}
LANGUAGES = {"English": "English", "తెలుగు · Telugu": "Telugu", "हिन्दी · Hindi": "Hindi", "தமிழ் · Tamil": "Tamil", "ಕನ್ನಡ · Kannada": "Kannada", "മലയാളം · Malayalam": "Malayalam"}
INTRODUCTIONS = {
    "English": "{name}, here's what I found in the agriculture documents:",
    "Telugu": "{name}, వ్యవసాయ పత్రాల్లో నేను కనుగొన్న సమాచారం ఇదిగో:",
    "Hindi": "{name}, कृषि दस्तावेज़ों में मुझे यह जानकारी मिली:",
    "Tamil": "{name}, விவசாய ஆவணங்களில் நான் கண்ட தகவல் இதோ:",
    "Kannada": "{name}, ಕೃಷಿ ದಾಖಲೆಗಳಲ್ಲಿ ನಾನು ಕಂಡ ಮಾಹಿತಿ ಇಲ್ಲಿದೆ:",
    "Malayalam": "{name}, കാർഷിക രേഖകളിൽ ഞാൻ കണ്ടെത്തിയ വിവരം ഇതാ:",
}
NO_PASSAGES = {
    "English": "I couldn't find relevant passages. Try a more specific question or index your PDFs first.",
    "Telugu": "సంబంధిత సమాచారం దొరకలేదు. మరింత స్పష్టమైన ప్రశ్న అడగండి లేదా ముందుగా మీ PDFలను సూచికలో చేర్చండి.",
    "Hindi": "संबंधित जानकारी नहीं मिली। अधिक स्पष्ट प्रश्न पूछें या पहले अपने PDF को अनुक्रमित करें।",
    "Tamil": "தொடர்புடைய தகவல் கிடைக்கவில்லை. தெளிவான கேள்வி கேளுங்கள் அல்லது முதலில் PDFகளை அட்டவணைப்படுத்துங்கள்.",
    "Kannada": "ಸಂಬಂಧಿತ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ. ಹೆಚ್ಚು ಸ್ಪಷ್ಟವಾದ ಪ್ರಶ್ನೆ ಕೇಳಿ ಅಥವಾ ಮೊದಲು PDFಗಳನ್ನು ಸೂಚ್ಯಂಕಕ್ಕೆ ಸೇರಿಸಿ.",
    "Malayalam": "ബന്ധപ്പെട്ട വിവരം കണ്ടെത്താനായില്ല. കൂടുതൽ വ്യക്തമായ ചോദ്യം ചോദിക്കുക അല്ലെങ്കിൽ ആദ്യം PDFകൾ സൂചികയിൽ ചേർക്കുക.",
}
FAILURES = {
    "English": "I couldn't complete this answer. Please try again.",
    "Telugu": "ఈ సమాధానాన్ని పూర్తి చేయలేకపోయాను. దయచేసి మళ్లీ ప్రయత్నించండి.",
    "Hindi": "मैं यह उत्तर पूरा नहीं कर सका। कृपया फिर से प्रयास करें।",
    "Tamil": "இந்த பதிலை முடிக்க முடியவில்லை. மீண்டும் முயற்சிக்கவும்.",
    "Kannada": "ಈ ಉತ್ತರವನ್ನು ಪೂರ್ಣಗೊಳಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "Malayalam": "ഈ ഉത്തരം പൂർത്തിയാക്കാനായില്ല. വീണ്ടും ശ്രമിക്കുക.",
}


@st.cache_resource
def load_backend():
    # Load the embedding model once, only when someone asks a question.
    from rag import retrieve, generate_answer, prepare_query, index_uploads

    return retrieve, generate_answer, prepare_query, index_uploads


def clear_conversation():
    st.session_state.messages = []
    st.session_state.pending_question = None


def queue_question(question):
    st.session_state.pending_question = question


def return_home():
    clear_conversation()
    st.session_state.user_name = ""
    st.session_state.uploaded_index = None
    st.session_state.upload_signature = None
    st.session_state.pop("pdf_uploads", None)


def personalized_answer(answer, language="English"):
    # Keep the name out of the retrieval query and model prompt.
    name = re.sub(r"([\\`*_{}\[\]()<>#+.!|~-])", r"\\\1", st.session_state.user_name)
    return INTRODUCTIONS[language].format(name=name) + "\n\n" + answer


def render_message(message):
    avatar = "🌱" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("error"):
            st.caption("Please try your question again. Check your connection and API configuration if this continues.")
        passages = message.get("sources", [])
        if passages:
            unique_sources = list(dict.fromkeys((c["source"], c["page"]) for c in passages))
            with st.expander(f"Sources · {len(unique_sources)} document pages"):
                st.caption("These passages were retrieved from the knowledge base for this answer.")
                for number, chunk in enumerate(passages, start=1):
                    title = DOCUMENT_NAMES.get(chunk["source"], chunk["source"])
                    st.write(f"**[{number}] {title} · Page {chunk['page']}**")
                    st.write(chunk["text"])
                    st.divider()
            with st.expander("Technical details"):
                st.write("Search query: " + message.get("search_query", ""))
                st.caption("Answer language: " + message.get("language", "English"))
                st.caption("Question → sentence embedding → FAISS search → retrieved passages → AI answer")
                for number, chunk in enumerate(passages, start=1):
                    st.caption(f"Passage {number}: FAISS distance {chunk['distance']:.4f} (lower means closer; this is not an answer confidence score).")


def export_conversation():
    lines = ["# Agriculture Assistant conversation", ""]
    for message in st.session_state.messages:
        label = "You" if message["role"] == "user" else "Assistant"
        lines.extend([f"## {label}", message["content"], ""])
        for number, chunk in enumerate(message.get("sources", []), start=1):
            lines.append(f"- [{number}] {DOCUMENT_NAMES.get(chunk['source'], chunk['source'])}, page {chunk['page']}: {chunk['text']}")
        lines.append("")
    return "\n".join(lines)


st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("user_name", "")
st.session_state.setdefault("uploaded_index", None)
st.session_state.setdefault("upload_signature", None)

background = base64.b64encode((Path(__file__).parent / "assets" / "fields.svg").read_bytes()).decode("ascii")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(90deg, rgba(250,252,248,.96), rgba(250,252,248,.85)), url("data:image/svg+xml;base64,{background}");
    background-size: cover; background-position: center; background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stChatMessage"], [data-testid="stForm"] {{ background: rgba(250,252,248,.92); }}
.welcome-hero {{ padding: clamp(1.5rem, 5vw, 4rem) 0 2rem; max-width: 650px; }}
.welcome-hero h1 {{ font-size: clamp(2.3rem, 6vw, 4rem); line-height: 1.12; letter-spacing: -.04em; }}
.welcome-hero p {{ color: #526359; font-size: 1.1rem; line-height: 1.7; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container { max-width: 1050px; padding-top: 2.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { border-right: 1px solid #dce6dd; }
.agri-header { border-left: 4px solid #34734b; padding: 0.25rem 0 0.25rem 1.2rem; margin-bottom: 1.6rem; }
.agri-eyebrow { color: #34734b; font-size: .78rem; font-weight: 700; letter-spacing: .12em; }
.agri-header h1 { font-size: clamp(1.8rem, 4vw, 2.6rem); padding: .4rem 0; margin: 0; }
.agri-header p { color: #526359; margin: .3rem 0; line-height: 1.6; }
[data-testid="stChatMessage"] { border: 1px solid #dce6dd; border-radius: 12px; }
.stButton button { border-radius: 9px; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.user_name:
    st.markdown("""
    <div class="welcome-hero">
      <div class="agri-eyebrow">🌱 AGRI · AGRICULTURE × TECHNOLOGY</div>
      <h1>A little curiosity.<br>A greener tomorrow.</h1>
      <p>Your companion for understanding soil, crops, water, and smarter farming.
      Explore agriculture knowledge with AI, backed by document sources.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.form("welcome_form", border=True):
        st.subheader("Let's get to know you")
        entered_name = st.text_input("What should we call you?", placeholder="Enter your name", max_chars=50, key="welcome_name")
        st.caption("Your name is used only in this browser session to personalise your experience.")
        submitted = st.form_submit_button("Start exploring →", type="primary", use_container_width=True)
        if submitted:
            clean_name = " ".join(entered_name.split())
            if not clean_name:
                st.warning("Please enter your name to get started.")
            elif any(ord(character) < 32 for character in clean_name):
                st.warning("Please enter a name without control characters.")
            else:
                st.session_state.user_name = clean_name
                st.rerun()
    st.caption("Soil health · Water management · Sustainable crops · Precision agriculture")
    st.stop()

with st.sidebar:
    st.title("🌱 Agri")
    st.caption("Agriculture knowledge, made accessible.")
    st.text(f"Welcome, {st.session_state.user_name}.")
    st.button("Change name / Home", on_click=return_home, use_container_width=True)
    selected_language = st.selectbox("Answer language", list(LANGUAGES), key="answer_language")
    language = LANGUAGES[selected_language]
    st.caption("Type in any supported language. New answers use your selected language.")
    st.button("＋ New conversation", on_click=clear_conversation, use_container_width=True)
    if st.session_state.messages:
        st.download_button("Download conversation", data=export_conversation(), file_name="agriculture-conversation.md", mime="text/markdown", use_container_width=True)
    st.divider()
    st.subheader("Knowledge base")
    st.caption("Answers use the indexed agriculture documents.")
    with st.expander("Browse document topics"):
        for title in DOCUMENT_NAMES.values():
            st.write(f"• {title}")
    with st.expander("Add your PDFs"):
        st.caption("Up to 5 English text PDFs, 10 MB each and 100 pages total. Scanned PDFs need OCR first.")
        uploads = st.file_uploader("Choose agriculture PDFs", type=["pdf"], accept_multiple_files=True, max_upload_size=10, key="pdf_uploads")
        files = [(upload.name, upload.getvalue()) for upload in uploads]
        signature = tuple((name, hashlib.sha256(data).hexdigest()) for name, data in files)
        if signature != st.session_state.upload_signature:
            st.session_state.uploaded_index = None
        st.caption("PDFs are held in this session. Retrieved passages are sent to OpenAI when you ask a question.")
        if st.button("Index selected PDFs", disabled=not files, use_container_width=True):
            try:
                with st.spinner("Reading PDFs and creating their search index…"):
                    _, _, _, index_uploads = load_backend()
                    indexed = index_uploads(files)
                st.session_state.uploaded_index = indexed
                st.session_state.upload_signature = signature
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                logging.getLogger(__name__).exception("PDF indexing failed")
                st.error("Couldn't index these PDFs. Please check the files and try again.")
        if st.session_state.uploaded_index:
            st.success(f"{len(files)} PDFs ready · {len(st.session_state.uploaded_index['chunks'])} passages")
        elif files:
            st.info("Click Index selected PDFs before searching them.")
    scope = st.selectbox("Search in", ["All documents", "Built-in documents only", "Uploaded PDFs only"], key="search_scope")
    st.divider()
    with st.expander("How it works"):
        st.write("1. Ask an agriculture question.\n2. Semantic search finds relevant document passages.\n3. AI uses those passages to write an answer.")
        st.caption("Powered by FAISS, Sentence Transformers, and OpenAI.")
    st.caption("Follow-up questions use the last three exchanges. Names, conversations and uploaded indexes last for this session.")

st.markdown(f"""
<div class="agri-header">
  <div class="agri-eyebrow">AGRICULTURE × TECHNOLOGY</div>
  <h1>Welcome, {html.escape(st.session_state.user_name)}.</h1>
  <p>Ask about soil, crops, water, and smarter farming.<br>Explore answers with the document passages behind them.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.subheader("Where would you like to start?")
    st.caption("Choose a topic to ask a starter question, or write your own below.")
    columns = st.columns(2)
    for position, (label, question) in enumerate(STARTERS.items()):
        with columns[position % 2]:
            with st.container(border=True):
                st.button(label, key=f"starter_{position}", on_click=queue_question, args=(question,), use_container_width=True)
                st.caption(question)
else:
    st.caption("Your conversation · Open Sources below an answer to inspect the evidence.")

for message in st.session_state.messages:
    render_message(message)

typed_question = st.chat_input("Ask an agriculture question…", key="question_input")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question is not None:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
    elif scope == "Uploaded PDFs only" and not st.session_state.uploaded_index:
        st.warning("Please index your PDFs first, or choose the built-in documents.")
    else:
        history = list(st.session_state.messages)
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        try:
            with st.status("Finding an answer…", expanded=True) as status:
                st.write("Understanding your question and its conversation context…")
                retrieve, generate_answer, prepare_query, _ = load_backend()
                search_query = prepare_query(question, history=history)
                st.write("Searching document passages…")
                passages = retrieve(search_query, top_k=4, uploaded=st.session_state.uploaded_index, scope=scope)
                if not passages:
                    answer = NO_PASSAGES[language]
                else:
                    st.write("Writing an answer from the retrieved passages…")
                    answer = generate_answer(question, passages, history=history, language=language)
                    if not answer or not answer.strip():
                        raise ValueError("The model returned an empty answer")
                status.update(label="Answer ready", state="complete", expanded=False)
            assistant_message = {"role": "assistant", "content": personalized_answer(answer, language), "raw_content": answer, "sources": passages, "search_query": search_query, "language": language}
        except Exception:
            logging.getLogger(__name__).exception("Agriculture question failed")
            assistant_message = {"role": "assistant", "content": FAILURES[language], "error": True, "language": language}
        st.session_state.messages.append(assistant_message)
        st.rerun()

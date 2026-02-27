import os
import streamlit as st
from huggingface_hub import InferenceClient


# ─────────────────────────────────────────────
#  Page Config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Qwen Chat · HuggingFace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS  – sleek dark terminal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

/* ── Root & Body ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080810 !important;
    color: #dde1f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stSidebar"] {
    background: #0e0e1a !important;
    border-right: 1px solid #1e1e30 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Custom header bar ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 22px 0;
    border-bottom: 1px solid #1e1e30;
    margin-bottom: 24px;
}
.chat-header .logo-pill {
    background: linear-gradient(135deg, #6c63ff, #00e5c0);
    border-radius: 12px;
    width: 44px; height: 44px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.chat-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important; padding: 0 !important;
    color: #fff !important;
    letter-spacing: -0.02em;
}
.chat-header .sub {
    font-size: 0.72rem;
    color: #5a5a80;
    margin-top: 2px;
}

/* ── Chat bubbles ── */
.msg-row { display: flex; gap: 12px; margin: 10px 0; align-items: flex-start; }
.msg-row.user  { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0; margin-top: 2px;
}
.avatar.bot  { background: #141428; border: 1px solid #2a2a45; }
.avatar.user { background: #1a0d35; border: 1px solid #3d2080; }

.bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 0.87rem;
    line-height: 1.7;
    position: relative;
}
.bubble.bot {
    background: #0f0f1e;
    border: 1px solid #1e1e35;
    border-top-left-radius: 4px;
    color: #dde1f0;
}
.bubble.user {
    background: #150d30;
    border: 1px solid #3a2070;
    border-top-right-radius: 4px;
    color: #e8e0ff;
    text-align: right;
}
.bubble .ts {
    font-size: 0.63rem;
    color: #3a3a60;
    margin-top: 6px;
    display: block;
}
.bubble.user .ts { text-align: right; }

/* ── Image preview inside bubble ── */
.img-preview {
    border-radius: 10px; overflow: hidden;
    border: 1px solid #2a2a45;
    margin-bottom: 8px; display: inline-block;
}
.img-preview img { max-width: 240px; max-height: 160px; display: block; object-fit: cover; }

/* ── Sidebar elements ── */
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 800;
    color: #fff; margin-bottom: 4px;
}
.sidebar-sub { font-size: 0.7rem; color: #555580; margin-bottom: 18px; }

.stat-box {
    background: #12121f;
    border: 1px solid #1e1e30;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.stat-box .s-label { font-size: 0.65rem; color: #555580; text-transform: uppercase; letter-spacing:.06em; }
.stat-box .s-value { font-size: 1rem; color: #00e5c0; font-weight: 600; margin-top: 2px; }

/* ── Input area tweaks ── */
[data-testid="stChatInput"] textarea {
    background: #0e0e1c !important;
    border: 1px solid #2a2a45 !important;
    color: #dde1f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,.2) !important;
}

/* ── Selectbox / text input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
    background: #12121f !important;
    border: 1px solid #2a2a45 !important;
    color: #dde1f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
[data-testid="baseButton-secondary"] {
    background: #12121f !important;
    border: 1px solid #2a2a45 !important;
    color: #dde1f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    transition: all .2s !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #6c63ff !important;
    color: #a89dff !important;
}
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #6c63ff, #4dd9b0) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0e0e1c !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6c63ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 4px; }

/* ── Token badge ── */
.model-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.68rem;
    color: #7070a0;
    margin-bottom: 16px;
}
.dot-online  { color: #00e5c0; }
.dot-offline { color: #f87171; }

/* ── Code blocks ── */
code { 
    background: #1a1a2e !important; 
    border: 1px solid #2a2a45 !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    color: #00e5c0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
pre code { padding: 14px !important; display: block; }

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #0e0e1e, #12102a);
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 40px auto;
    max-width: 520px;
}
.welcome-card .wc-icon { font-size: 48px; margin-bottom: 12px; }
.welcome-card h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem; font-weight: 800;
    color: #fff; margin-bottom: 8px;
}
.welcome-card p { font-size: 0.8rem; color: #555580; line-height: 1.6; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
.chip {
    background: #12121f; border: 1px solid #2a2a45;
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.72rem; color: #7070a0;
    cursor: default;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Constants & Models
# ─────────────────────────────────────────────
MODELS = {
    "Qwen 3.5 397B · Novita  🔥": "Qwen/Qwen3.5-397B-A17B:novita",
    "Qwen 2.5 72B Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.3 70B Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta",
}

SYSTEM_PROMPTS = {
    "🤖 General Assistant": "You are a highly capable, professional AI assistant. Be concise, accurate, and helpful. Format code with markdown code blocks.",
    "🧑‍💻 Code Expert": "You are an expert software engineer. Provide clean, well-commented code. Always explain your implementation. Prefer modern best practices.",
    "🔬 Research Analyst": "You are a rigorous research analyst. Provide thorough, well-sourced analysis. Structure your responses with clear sections.",
    "✍️ Creative Writer": "You are a creative writer with a vivid, engaging style. Help with storytelling, copywriting, poetry, and all forms of creative writing.",
    "📊 Data Scientist": "You are an expert data scientist. Help with statistics, ML models, data analysis, visualization, and Python/R code.",
}


# ─────────────────────────────────────────────
#  Session State Init
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_msgs" not in st.session_state:
    st.session_state.total_msgs = 0
if "model_key" not in st.session_state:
    st.session_state.model_key = list(MODELS.keys())[0]
if "system_key" not in st.session_state:
    st.session_state.system_key = list(SYSTEM_PROMPTS.keys())[0]
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 1024


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">HuggingFace InferenceClient</div>', unsafe_allow_html=True)

    # Token status
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        st.markdown('<span class="dot-online">●</span> <span style="font-size:.75rem;color:#aaa"> HF_TOKEN set</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot-offline">●</span> <span style="font-size:.75rem;color:#f87171"> HF_TOKEN missing</span>', unsafe_allow_html=True)
        st.warning("Set HF_TOKEN env var:\n```\nexport HF_TOKEN=hf_...\n```", icon="⚠️")

    st.divider()

    # Model picker
    st.markdown("**Model**")
    model_key = st.selectbox(
        "model",
        options=list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    model_id = MODELS[model_key]
    st.markdown(f'<div class="model-badge">{model_id}</div>', unsafe_allow_html=True)

    # System prompt
    st.markdown("**Persona**")
    system_key = st.selectbox(
        "persona",
        options=list(SYSTEM_PROMPTS.keys()),
        label_visibility="collapsed",
    )
    system_prompt = SYSTEM_PROMPTS[system_key]

    st.divider()

    # Parameters
    st.markdown("**Parameters**")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05,
        help="Higher = more creative, lower = more deterministic")
    max_tokens = st.slider("Max tokens", 128, 4096, 1024, 128)
    stream_mode = st.toggle("⚡ Streaming", value=True)

    st.divider()

    # Image URL input
    st.markdown("**🖼 Image URL** *(optional)*")
    image_url = st.text_input(
        "img_url",
        placeholder="https://example.com/image.jpg",
        label_visibility="collapsed",
    )
    if image_url:
        st.image(image_url, use_container_width=True)

    st.divider()

    # Stats
    st.markdown("**Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="s-label">Messages</div>
            <div class="s-value">{st.session_state.total_msgs}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="s-label">Turns</div>
            <div class="s-value">{len(st.session_state.messages) // 2}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_msgs = 0
        st.rerun()

    # Export chat
    if st.session_state.messages:
        export_text = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content'] if isinstance(m['content'], str) else m['content'][0]['text']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "💾 Export chat",
            data=export_text,
            file_name="chat_export.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ─────────────────────────────────────────────
#  Main Chat Area
# ─────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <div class="logo-pill">🤖</div>
  <div>
    <h1>Qwen Chat</h1>
    <div class="sub">HuggingFace InferenceClient · Multimodal · Streaming</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Welcome state
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
      <div class="wc-icon">✨</div>
      <h2>Start a conversation</h2>
      <p>Powered by Qwen 3.5 397B via HuggingFace InferenceClient.<br>
         Supports text, code, and image URL analysis.</p>
      <div class="chip-row">
        <span class="chip">🧠 Reasoning</span>
        <span class="chip">💻 Code</span>
        <span class="chip">🖼 Vision</span>
        <span class="chip">✍️ Writing</span>
        <span class="chip">📊 Analysis</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Render conversation history
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    # Extract text and image_url from content
    if isinstance(content, list):
        text_parts = [c["text"] for c in content if c["type"] == "text"]
        img_parts  = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
        text = " ".join(text_parts)
        img  = img_parts[0] if img_parts else None
    else:
        text = content
        img  = None

    with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
        if img:
            st.image(img, width=260)
        st.markdown(text)


# ─────────────────────────────────────────────
#  Chat Input
# ─────────────────────────────────────────────
user_input = st.chat_input("Message Qwen… (attach image URL in sidebar)")

if user_input:
    if not hf_token:
        st.error("⚠️ Please set your HF_TOKEN environment variable first.")
        st.stop()

    # Build user message content
    if image_url:
        user_content = [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        display_img = image_url
    else:
        user_content = user_input
        display_img = None

    # Add to history & display
    st.session_state.messages.append({"role": "user", "content": user_content})
    st.session_state.total_msgs += 1

    with st.chat_message("user", avatar="👤"):
        if display_img:
            st.image(display_img, width=260)
        st.markdown(user_input)

    # ── Build API messages ──
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # ── Call HF InferenceClient ──
    client = InferenceClient(api_key=hf_token)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""

        try:
            if stream_mode:
                with st.spinner(""):
                    stream = client.chat.completions.create(
                        model=model_id,
                        messages=api_messages,
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            else:
                with st.spinner("Thinking…"):
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=api_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    full_response = completion.choices[0].message.content
                placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"❌ **Error:** `{e}`"
            placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.total_msgs += 1
    st.rerun()
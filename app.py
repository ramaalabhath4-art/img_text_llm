import os
import io
import base64
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

st.set_page_config(
    page_title="Qwen Studio - HuggingFace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:       #05050f;
  --surface:  #0a0a18;
  --surface2: #0f0f22;
  --border:   #1c1c38;
  --border2:  #2a2a52;
  --accent:   #7c6dfa;
  --pink:     #fa6d9a;
  --green:    #2fffa0;
  --cyan:     #2fe8ff;
  --text:     #e8e8f8;
  --muted:    #5a5a82;
  --r:        14px;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
#MainMenu, footer { visibility: hidden; }

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 900px 700px at 15% 5%,  rgba(124,109,250,.08) 0%, transparent 70%),
    radial-gradient(ellipse 700px 900px at 85% 95%, rgba(250,109,154,.06) 0%, transparent 70%),
    radial-gradient(ellipse 500px 500px at 50% 50%, rgba(47,255,160,.03)  0%, transparent 60%);
}

.app-bar {
  display: flex; align-items: center; gap: 14px;
  padding: 18px 0 20px; border-bottom: 1px solid var(--border);
}
.logo-ring {
  width: 46px; height: 46px; border-radius: 14px;
  background: linear-gradient(135deg, var(--accent), var(--pink));
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; box-shadow: 0 0 28px rgba(124,109,250,.4);
}
.app-title {
  font-family: 'Outfit', sans-serif;
  font-size: 1.45rem; font-weight: 900; color: #fff; letter-spacing: -.035em;
}
.app-title em { font-style: normal; color: var(--accent); }
.app-sub { font-size: .7rem; color: var(--muted); margin-top: 2px; }

[data-testid="stChatMessage"] {
  background: transparent !important; border: none !important; padding: 3px 0 !important;
}
[data-testid="stChatMessageContent"] {
  background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r) !important; padding: 14px 18px !important;
  font-size: .9rem !important; line-height: 1.72 !important; color: var(--text) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
  [data-testid="stChatMessageContent"] {
  background: #110d2a !important; border-color: #2e2062 !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; font-family: 'Outfit',sans-serif !important;
  border-radius: 10px !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; border-radius: 10px !important;
}

[data-testid="baseButton-secondary"] {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; font-family: 'Outfit',sans-serif !important;
  border-radius: 10px !important; font-size: .83rem !important; font-weight: 500 !important;
  transition: all .2s !important;
}
[data-testid="baseButton-secondary"]:hover {
  border-color: var(--accent) !important; color: #c4bcff !important;
}
[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--accent), #5a4ed4) !important;
  border: none !important; color: #fff !important;
  font-family: 'Outfit',sans-serif !important; border-radius: 10px !important;
  font-weight: 700 !important;
}

.sb-brand { font-family:'Outfit',sans-serif; font-size:1.05rem; font-weight:900; color:#fff; margin-bottom:2px; }
.sb-sub   { font-size:.68rem; color:var(--muted); margin-bottom:14px; }
.tok-ok   { color:var(--green); font-size:.78rem; font-weight:600; }
.tok-bad  { color:var(--pink);  font-size:.78rem; font-weight:600; }

.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin:6px 0; }
.stat-card { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
.stat-card .sl { font-size:.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }
.stat-card .sv { font-size:1.05rem; font-weight:700; color:var(--green); margin-top:2px; }
.model-tag {
  display:inline-block; margin:4px 0 10px;
  background:var(--surface2); border:1px solid var(--border2);
  border-radius:20px; padding:3px 10px;
  font-size:.65rem; color:var(--muted); font-family:'JetBrains Mono',monospace;
}

.sec-card { background:var(--surface2); border:1px solid var(--border); border-radius:var(--r); padding:22px 26px; margin-bottom:18px; }
.sec-title { font-family:'Outfit',sans-serif; font-size:1.05rem; font-weight:700; color:#fff; margin-bottom:4px; display:flex; align-items:center; gap:10px; }
.sec-sub   { font-size:.75rem; color:var(--muted); margin-bottom:16px; }
.badge-v { background:rgba(124,109,250,.14);border:1px solid rgba(124,109,250,.3);color:#c4bcff;border-radius:6px;padding:2px 8px;font-size:.67rem;font-weight:600; }
.badge-p { background:rgba(250,109,154,.12);border:1px solid rgba(250,109,154,.3);color:#ffb3cc;border-radius:6px;padding:2px 8px;font-size:.67rem;font-weight:600; }
.badge-c { background:rgba(47,232,255,.1);  border:1px solid rgba(47,232,255,.25);color:#a8f5ff;border-radius:6px;padding:2px 8px;font-size:.67rem;font-weight:600; }

.welcome { display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:44vh;text-align:center;padding:40px 20px; }
.w-glyph { font-size:52px;margin-bottom:14px; }
.w-title { font-family:'Outfit',sans-serif; font-size:1.9rem; font-weight:900; color:#fff; letter-spacing:-.04em; margin-bottom:10px; }
.w-title em { font-style:normal; color:var(--accent); }
.w-desc  { font-size:.82rem; color:var(--muted); max-width:420px; line-height:1.65; margin-bottom:22px; }
.w-chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; }
.w-chip  { background:var(--surface2); border:1px solid var(--border2); border-radius:20px; padding:6px 16px; font-size:.75rem; color:var(--muted); font-weight:500; }

.result-box { background:var(--surface2); border:1px solid var(--border); border-radius:var(--r); padding:20px 24px; font-size:.9rem; line-height:1.72; }

::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:4px; }

hr { border-color:var(--border) !important; }
code {
  background:#131328 !important; border:1px solid var(--border2) !important;
  color:var(--green) !important; border-radius:5px !important;
  font-family:'JetBrains Mono',monospace !important; font-size:.82em !important;
  padding:2px 6px !important;
}
[data-testid="stFileUploader"] { background:var(--surface2) !important; border:1px dashed var(--border2) !important; border-radius:var(--r) !important; }
[data-testid="stExpander"]     { background:var(--surface2) !important; border:1px solid var(--border) !important; border-radius:10px !important; }

/* Hide native Streamlit bottom input bar - replaced by custom voice bar */
[data-testid="stBottom"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHAT_MODELS = {
    "Qwen 3.5 397B - Novita":    "Qwen/Qwen3.5-397B-A17B:novita",
    "Qwen 2.5 72B Instruct":     "Qwen/Qwen2.5-72B-Instruct",
    "Qwen 2.5 VL 72B (Vision)":  "Qwen/Qwen2.5-VL-72B-Instruct",
    "Llama 3.3 70B Instruct":    "meta-llama/Llama-3.3-70B-Instruct",
    "Llama 3.2 11B Vision":      "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Mistral 7B Instruct":       "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr 7B Beta":            "HuggingFaceH4/zephyr-7b-beta",
}
T2I_MODELS = {
    "FLUX.1 Schnell (fast)":     "black-forest-labs/FLUX.1-schnell",
    "FLUX.1 Dev (quality)":      "black-forest-labs/FLUX.1-dev",
    "Stable Diffusion 3.5":      "stabilityai/stable-diffusion-3.5-large",
    "SD XL Base 1.0":            "stabilityai/stable-diffusion-xl-base-1.0",
}
I2T_MODELS = {
    "Qwen 2.5 VL 72B":           "Qwen/Qwen2.5-VL-72B-Instruct",
    "LLaVA 1.6 Mistral 7B":      "llava-hf/llava-v1.6-mistral-7b-hf",
    "InternVL2 8B":              "OpenGVLab/InternVL2-8B",
}
PERSONAS = {
    "General Assistant":  "You are a highly capable, professional AI assistant. Be concise, accurate, and helpful. Format code with markdown code blocks.",
    "Code Expert":        "You are an expert software engineer. Write clean, well-commented, modern code. Always explain your implementation.",
    "Research Analyst":   "You are a rigorous research analyst. Provide thorough, well-structured analysis with clear headings.",
    "Creative Writer":    "You are a creative writer with a vivid, engaging style. Help with storytelling, copywriting, and creative writing.",
    "Data Scientist":     "You are an expert data scientist. Help with statistics, ML models, data analysis, and Python/R code.",
}
VISION_MODELS = {
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "OpenGVLab/InternVL2-8B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "messages":   [],
    "total_msgs": 0,
    "images_gen": 0,
    "active_tab": "Chat",
    "gen_image":  None,
    "i2t_result": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

hf_token = os.environ.get("HF_TOKEN", "")

def get_client():
    return InferenceClient(api_key=hf_token)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">Qwen Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">HuggingFace InferenceClient</div>', unsafe_allow_html=True)

    if hf_token:
        st.markdown('<div class="tok-ok">&#9679; HF_TOKEN loaded OK</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tok-bad">&#9679; HF_TOKEN missing</div>', unsafe_allow_html=True)
        with st.expander("Fix token issue", expanded=True):
            st.markdown("""
Create a `.env` file next to `app.py`:
```
HF_TOKEN=hf_your_token_here
```
Get token at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
Then restart: `streamlit run app.py`
""")

    st.divider()
    tab = st.session_state.active_tab
    st.markdown(f"**Active Mode:** `{tab}`")
    st.divider()

    # Default sidebar values
    image_url     = None
    chat_model_id = list(CHAT_MODELS.values())[0]
    system_prompt = list(PERSONAS.values())[0]
    temperature   = 0.7
    max_tokens    = 1024
    stream_mode   = True
    t2i_model     = list(T2I_MODELS.values())[0]
    img_w = img_h = 1024
    t2i_steps     = 4
    i2t_model     = list(I2T_MODELS.values())[0]

    if tab == "Chat":
        st.markdown("**Model**")
        ck = st.selectbox("chat_m", list(CHAT_MODELS.keys()), label_visibility="collapsed")
        chat_model_id = CHAT_MODELS[ck]
        st.markdown(f'<div class="model-tag">{chat_model_id}</div>', unsafe_allow_html=True)
        st.markdown("**Persona**")
        pk = st.selectbox("persona", list(PERSONAS.keys()), label_visibility="collapsed")
        system_prompt = PERSONAS[pk]
        st.markdown("**Parameters**")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens  = st.slider("Max tokens", 128, 4096, 1024, 128)
        stream_mode = st.toggle("Streaming", value=True)
        st.divider()
        st.markdown("**Image URL (optional)**")
        image_url = st.text_input("iurl", placeholder="https://example.com/image.jpg",
                                  label_visibility="collapsed")
        if image_url:
            st.image(image_url, use_container_width=True)

    elif tab == "Text to Image":
        st.markdown("**Image Model**")
        tk = st.selectbox("t2i_m", list(T2I_MODELS.keys()), label_visibility="collapsed")
        t2i_model = T2I_MODELS[tk]
        st.markdown(f'<div class="model-tag">{t2i_model}</div>', unsafe_allow_html=True)
        st.markdown("**Canvas Size**")
        img_w     = st.select_slider("Width",  [512,768,1024,1280], 1024)
        img_h     = st.select_slider("Height", [512,768,1024,1280], 1024)
        t2i_steps = st.slider("Steps", 1, 50, 4, 1)

    elif tab == "Image to Text":
        st.markdown("**Vision Model**")
        ik = st.selectbox("i2t_m", list(I2T_MODELS.keys()), label_visibility="collapsed")
        i2t_model = I2T_MODELS[ik]
        st.markdown(f'<div class="model-tag">{i2t_model}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Session**")
    st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card"><div class="sl">Messages</div><div class="sv">{st.session_state.total_msgs}</div></div>
  <div class="stat-card"><div class="sl">Turns</div><div class="sv">{len(st.session_state.messages)//2}</div></div>
  <div class="stat-card"><div class="sl">Images</div><div class="sv">{st.session_state.images_gen}</div></div>
  <div class="stat-card"><div class="sl">Mode</div><div class="sv" style="font-size:.7rem;color:#9090c0">{tab}</div></div>
</div>""", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear", use_container_width=True):
            st.session_state.update(messages=[], total_msgs=0, gen_image=None, i2t_result="")
            st.rerun()
    with c2:
        if st.session_state.messages:
            export = "\n\n".join(
                f"[{m['role'].upper()}]\n{m['content'] if isinstance(m['content'],str) else m['content'][0]['text']}"
                for m in st.session_state.messages
            )
            st.download_button("Export", data=export, file_name="chat.txt",
                               mime="text/plain", use_container_width=True)

# ── App Bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-bar">
  <div class="logo-ring">&#x1F916;</div>
  <div>
    <div class="app-title">Qwen <em>Studio</em></div>
    <div class="app-sub">HuggingFace InferenceClient &middot; Multimodal &middot; Streaming &middot; Voice</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ── Tab navigation ─────────────────────────────────────────────────────────────
TABS = ["Chat", "Text to Image", "Image to Text"]
TAB_LABELS = ["💬 Chat", "🎨 Text → Image", "🔍 Image → Text"]
cols = st.columns(len(TABS) + 5)
for i, (t, label) in enumerate(zip(TABS, TAB_LABELS)):
    with cols[i]:
        is_active = st.session_state.active_tab == t
        if st.button(label, key=f"nav_{i}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.active_tab = t
            st.rerun()

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 - CHAT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "Chat":

    if not st.session_state.messages:
        st.markdown("""
<div class="welcome">
  <div class="w-glyph">&#x2728;</div>
  <div class="w-title">Ask <em>anything</em></div>
  <div class="w-desc">
    Powered by Qwen 3.5 397B via HuggingFace InferenceClient.<br>
    Real-time streaming &middot; Multimodal vision &middot; Voice input &middot; TTS
  </div>
  <div class="w-chips">
    <span class="w-chip">&#x1F9E0; Reasoning</span>
    <span class="w-chip">&#x1F4BB; Code</span>
    <span class="w-chip">&#x1F5BC;&#xFE0F; Vision</span>
    <span class="w-chip">&#x270D;&#xFE0F; Writing</span>
    <span class="w-chip">&#x1F4CA; Analysis</span>
    <span class="w-chip">&#x1F3A4; Voice</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Render message history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            text = " ".join(c["text"] for c in content if c["type"] == "text")
            imgs = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
        else:
            text, imgs = content, []
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "🧑"):
            for img in imgs:
                st.image(img, width=280)
            st.markdown(text)

    # Vision model check
    model_supports_vision = chat_model_id in VISION_MODELS
    img_url = image_url
    if img_url and not model_supports_vision:
        st.warning(
            f"**{chat_model_id.split('/')[-1]}** does not support image input. "
            "Switch to **Qwen 2.5 VL 72B (Vision)** or remove the image URL.",
            icon="🖼",
        )

    # ── Unified Voice + Text Input Bar ───────────────────────────────────────
    components.html("""<!DOCTYPE html>
<html>
<head>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:transparent; font-family:'Outfit','Segoe UI',sans-serif; }
.bar {
  display:flex; align-items:center; gap:10px;
  padding:10px 14px;
  background:#0f0f22; border:1.5px solid #2a2a52; border-radius:16px;
  transition:border-color .2s;
}
.bar:focus-within { border-color:#7c6dfa; box-shadow:0 0 0 3px rgba(124,109,250,.12); }
.mic {
  width:40px; height:40px; border-radius:50%; flex-shrink:0;
  background:linear-gradient(135deg,#7c6dfa,#5a4ed4);
  border:none; cursor:pointer; font-size:18px;
  box-shadow:0 0 16px rgba(124,109,250,.4); transition:all .2s;
  display:flex; align-items:center; justify-content:center;
}
.mic:hover { transform:scale(1.08); }
.mic.rec {
  background:linear-gradient(135deg,#fa4060,#b0102e) !important;
  animation:pulse 1.1s infinite;
}
@keyframes pulse {
  0%  { box-shadow:0 0 0 0   rgba(250,60,80,.5); }
  70% { box-shadow:0 0 0 12px rgba(250,60,80,0); }
  100%{ box-shadow:0 0 0 0   rgba(250,60,80,0); }
}
.txt {
  flex:1; background:transparent; border:none; outline:none;
  color:#e8e8f8; font-family:inherit; font-size:.9rem;
  resize:none; min-height:24px; max-height:120px; line-height:1.5;
}
.txt::placeholder { color:#3a3a62; }
.div  { width:1px; height:26px; background:#2a2a52; flex-shrink:0; }
.lang {
  background:#0a0a18; border:1px solid #2a2a52; color:#5a5a82;
  border-radius:7px; font-size:.7rem; padding:5px 7px;
  cursor:pointer; font-family:inherit; flex-shrink:0;
}
.tts {
  padding:6px 11px; border-radius:8px; border:1px solid #2a2a52;
  background:#0a0a18; color:#5a5a82; font-size:.72rem; cursor:pointer;
  transition:all .2s; white-space:nowrap; font-family:inherit; flex-shrink:0;
}
.tts:hover { border-color:#2fe8ff; color:#2fe8ff; }
.tts.on    { border-color:#2fe8ff; color:#2fe8ff; background:rgba(47,232,255,.08); }
.send {
  width:38px; height:38px; border-radius:10px; flex-shrink:0;
  background:linear-gradient(135deg,#7c6dfa,#5a4ed4);
  border:none; color:#fff; font-size:17px; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 2px 12px rgba(124,109,250,.35); transition:all .2s;
}
.send:hover { transform:translateY(-1px); box-shadow:0 4px 18px rgba(124,109,250,.5); }
</style>
</head>
<body>
<div class="bar">
  <button class="mic" id="mic" title="Click to speak">&#x1F3A4;</button>
  <textarea class="txt" id="txt" rows="1"
    placeholder="Message Qwen... or click mic to speak"></textarea>
  <div class="div"></div>
  <select class="lang" id="lang">
    <option value="en-US">EN</option>
    <option value="fr-FR">FR</option>
    <option value="de-DE">DE</option>
    <option value="es-ES">ES</option>
    <option value="ar-SA">AR</option>
    <option value="zh-CN">ZH</option>
    <option value="ja-JP">JP</option>
  </select>
  <button class="tts on" id="tts" title="Toggle AI voice reply">&#x1F50A; TTS</button>
  <button class="send" id="send" title="Send (Enter)">&#x27A4;</button>
</div>
<script>
const mic=document.getElementById('mic'), txt=document.getElementById('txt'),
      lang=document.getElementById('lang'), tts=document.getElementById('tts'),
      send=document.getElementById('send');
let isRec=false, ttsOn=true, lastBot='';

txt.addEventListener('input',()=>{
  txt.style.height='auto';
  txt.style.height=Math.min(txt.scrollHeight,120)+'px';
});
txt.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend();}
});
send.addEventListener('click',doSend);

function doSend(){
  const msg=txt.value.trim(); if(!msg) return;
  const stIn=window.parent.document.querySelector('[data-testid="stChatInput"] textarea');
  if(stIn){
    const setter=Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value').set;
    setter.call(stIn,msg);
    stIn.dispatchEvent(new Event('input',{bubbles:true}));
    setTimeout(()=>stIn.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',keyCode:13,bubbles:true})),80);
  }
  txt.value=''; txt.style.height='auto';
}

const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!SR){
  mic.innerHTML='&#x1F6AB;'; mic.style.opacity='.4'; mic.style.cursor='default';
  mic.title='Use Chrome or Edge for voice input';
} else {
  const rec=new SR(); rec.continuous=false; rec.interimResults=true;
  rec.onstart=()=>{ isRec=true; mic.classList.add('rec'); mic.innerHTML='&#x23F9;&#xFE0F;'; txt.placeholder='Listening...'; };
  rec.onresult=e=>{
    let out='';
    for(let i=e.resultIndex;i<e.results.length;i++) out+=e.results[i][0].transcript;
    txt.value=out; txt.style.height='auto'; txt.style.height=Math.min(txt.scrollHeight,120)+'px';
  };
  rec.onend=()=>{ isRec=false; mic.classList.remove('rec'); mic.innerHTML='&#x1F3A4;'; txt.placeholder='Message Qwen... or click mic to speak'; };
  rec.onerror=e=>{ isRec=false; mic.classList.remove('rec'); mic.innerHTML='&#x1F3A4;'; txt.placeholder='Error: '+e.error+' - try again'; };
  mic.addEventListener('click',()=>{ if(isRec) rec.stop(); else{ rec.lang=lang.value; rec.start(); } });
}

tts.addEventListener('click',()=>{
  ttsOn=!ttsOn;
  tts.innerHTML=ttsOn?'&#x1F50A; TTS':'&#x1F507; TTS';
  tts.className='tts'+(ttsOn?' on':'');
  if(!ttsOn) window.speechSynthesis.cancel();
});

function speakLatest(){
  if(!ttsOn) return;
  const msgs=window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
  for(let i=msgs.length-1;i>=0;i--){
    if(msgs[i].querySelector('[data-testid="chatAvatarIcon-assistant"]')){
      const el=msgs[i].querySelector('[data-testid="stChatMessageContent"]');
      const t=el?.innerText?.trim();
      if(t&&t!==lastBot&&t.length<1800){
        lastBot=t;
        const u=new SpeechSynthesisUtterance(t);
        u.lang=lang.value; u.rate=1.05;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
      }
      break;
    }
  }
}
setInterval(speakLatest,1800);
</script>
</body>
</html>""", height=72, scrolling=False)

    # Hidden native input - still needed for Streamlit to process submissions
    user_input = st.chat_input("Message Qwen...", key="chat_hidden")

    if user_input:
        if not hf_token:
            st.error("HF_TOKEN not found. Add it to your .env file and restart.")
            st.stop()

        if img_url and model_supports_vision:
            user_content = [
                {"type": "text",      "text": user_input},
                {"type": "image_url", "image_url": {"url": img_url}},
            ]
            display_img = img_url
        else:
            user_content = user_input
            display_img  = None

        st.session_state.messages.append({"role": "user", "content": user_content})
        st.session_state.total_msgs += 1

        with st.chat_message("user", avatar="🧑"):
            if display_img:
                st.image(display_img, width=280)
            st.markdown(user_input)

        api_msgs = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages:
            content = m["content"]
            if isinstance(content, list) and not model_supports_vision:
                content = " ".join(c["text"] for c in content if c.get("type") == "text")
            api_msgs.append({"role": m["role"], "content": content})

        client = get_client()
        with st.chat_message("assistant", avatar="🤖"):
            box = st.empty()
            reply = ""
            try:
                if stream_mode:
                    stream = client.chat.completions.create(
                        model=chat_model_id, messages=api_msgs,
                        stream=True, max_tokens=max_tokens, temperature=temperature,
                    )
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            reply += delta
                            box.markdown(reply + "▌")
                    box.markdown(reply if reply else "_No response. Try a different model._")
                else:
                    with st.spinner("Thinking..."):
                        comp = client.chat.completions.create(
                            model=chat_model_id, messages=api_msgs,
                            max_tokens=max_tokens, temperature=temperature,
                        )
                    reply = comp.choices[0].message.content if comp.choices else "No response."
                    box.markdown(reply)
            except Exception as e:
                err = str(e)
                if "list index out of range" in err:
                    reply = ("**Model returned empty response.**\n\n"
                             "- Text-only model: remove the image URL\n"
                             "- For vision: switch to **Qwen 2.5 VL 72B**\n"
                             "- Try again — may be a timeout")
                else:
                    reply = f"**Error:** `{err}`"
                box.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.total_msgs += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 - TEXT TO IMAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "Text to Image":

    st.markdown("""
<div class="sec-card">
  <div class="sec-title">&#x1F3A8; Text to Image Generation <span class="badge-p">FLUX &middot; SD3.5 &middot; SDXL</span></div>
  <div class="sec-sub">Describe what you want to see and the model generates it instantly.</div>
</div>""", unsafe_allow_html=True)

    prompt = st.text_area("Image Prompt",
        placeholder="A cinematic photo of a futuristic Tokyo street at night, neon lights reflecting on wet pavement, 8K ultra-detailed...",
        height=120, key="t2i_prompt")

    neg_prompt = st.text_input("Negative Prompt (what to avoid)",
        placeholder="blurry, ugly, distorted, watermark, low quality...", key="t2i_neg")

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        gen_btn = st.button("Generate Image", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Model: `{t2i_model.split('/')[-1]}` · {img_w}x{img_h}px · {t2i_steps} steps")

    if gen_btn:
        if not hf_token:
            st.error("HF_TOKEN not found.")
        elif not (prompt or "").strip():
            st.warning("Please enter a prompt first.")
        else:
            prog = st.progress(0, text="Sending to model...")
            try:
                prog.progress(30, text="Generating...")
                client = get_client()
                kwargs = dict(prompt=prompt, width=img_w, height=img_h, num_inference_steps=t2i_steps)
                if neg_prompt:
                    kwargs["negative_prompt"] = neg_prompt
                image = client.text_to_image(model=t2i_model, **kwargs)
                prog.progress(100, text="Done!")
                st.session_state.gen_image  = image
                st.session_state.images_gen += 1
            except Exception as e:
                prog.empty()
                st.error(f"Error: `{e}`")

    if st.session_state.gen_image is not None:
        st.divider()
        col_img, col_act = st.columns([3, 1])
        with col_img:
            st.image(st.session_state.gen_image,
                     caption=f"Generated · {img_w}x{img_h} · {t2i_model.split('/')[-1]}",
                     use_container_width=True)
        with col_act:
            st.markdown("**Actions**")
            buf = io.BytesIO()
            st.session_state.gen_image.save(buf, format="PNG")
            st.download_button("Download PNG", data=buf.getvalue(),
                               file_name="generated.png", mime="image/png",
                               use_container_width=True)
            if st.button("Discard", use_container_width=True):
                st.session_state.gen_image = None
                st.rerun()
            if st.button("Discuss in Chat", use_container_width=True):
                st.session_state.active_tab = "Chat"
                st.rerun()
    else:
        st.divider()
        st.markdown("**Prompt Ideas**")
        ideas = [
            "A lone astronaut on a rust-red Martian desert at dawn, cinematic",
            "Hyperrealistic portrait of a silver android with glowing blue eyes, studio lighting",
            "Enchanted forest at midnight, bioluminescent mushrooms and fireflies, magical",
            "Aerial cyberpunk megacity at night, drone shot, neon lights, 8K ultra-detailed",
        ]
        c1, c2 = st.columns(2)
        for i, idea in enumerate(ideas):
            with (c1 if i % 2 == 0 else c2):
                if st.button(f"{idea[:52]}...", key=f"idea_{i}", use_container_width=True):
                    st.session_state["t2i_prompt"] = idea
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 - IMAGE TO TEXT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "Image to Text":

    st.markdown("""
<div class="sec-card">
  <div class="sec-title">&#x1F50D; Image to Text Analysis <span class="badge-c">Vision &middot; VL Models</span></div>
  <div class="sec-sub">Upload an image or provide a URL, then ask anything about it.</div>
</div>""", unsafe_allow_html=True)

    src_upload, src_url = st.tabs(["Upload File", "Image URL"])
    image_data_url = None

    with src_upload:
        uploaded = st.file_uploader("Drop an image", type=["png","jpg","jpeg","webp","gif"],
                                    label_visibility="collapsed")
        if uploaded:
            b64 = base64.b64encode(uploaded.read()).decode()
            image_data_url = f"data:{uploaded.type};base64,{b64}"
            st.image(uploaded, width=380)

    with src_url:
        url_in = st.text_input("Image URL", placeholder="https://example.com/image.jpg",
                               label_visibility="collapsed", key="i2t_url")
        if url_in:
            image_data_url = url_in
            st.image(url_in, width=380)

    st.markdown("**What do you want to know?**")
    qa, qb, qc, qd = st.columns(4)
    quick = {
        "Full Description":  "Describe this image in full detail.",
        "Extract Text (OCR)":"Extract and transcribe all text visible in this image.",
        "List Objects":      "List every object, person, and element in this image with their positions.",
        "Analyze Style":     "Analyze the visual style, colors, mood, and composition of this image.",
    }
    for (label, qp), col in zip(quick.items(), [qa, qb, qc, qd]):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["i2t_prompt"] = qp
                st.rerun()

    analysis_prompt = st.text_area("Custom question",
        value=st.session_state.get("i2t_prompt", "Describe this image in detail."),
        height=90, label_visibility="collapsed", key="i2t_prompt")

    if st.button("Analyze Image", type="primary"):
        if not hf_token:
            st.error("HF_TOKEN not found.")
        elif not image_data_url:
            st.warning("Please upload an image or enter a URL.")
        else:
            with st.spinner(f"Analyzing with {i2t_model.split('/')[-1]}..."):
                try:
                    client = get_client()
                    comp = client.chat.completions.create(
                        model=i2t_model,
                        messages=[{"role": "user", "content": [
                            {"type": "text",      "text": analysis_prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]}],
                        max_tokens=1024,
                    )
                    st.session_state.i2t_result = comp.choices[0].message.content
                except Exception as e:
                    st.session_state.i2t_result = f"Error: `{e}`"

    if st.session_state.i2t_result:
        st.divider()
        st.markdown("**Analysis Result**")
        st.markdown(
            f'<div class="result-box">{st.session_state.i2t_result}</div>',
            unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.download_button("Save Result", data=st.session_state.i2t_result,
                               file_name="analysis.txt", mime="text/plain",
                               use_container_width=True)
        with rc2:
            if st.button("Continue in Chat", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Image Analysis:**\n\n{st.session_state.i2t_result}",
                })
                st.session_state.active_tab = "Chat"
                st.rerun()
        with rc3:
            if st.button("New Analysis", use_container_width=True):
                st.session_state.i2t_result = ""
                st.rerun()

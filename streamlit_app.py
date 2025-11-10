# streamlit_app.py
# App RAG su indice Chroma già esistente (SOLO LETTURA)
# - MMR + keyword boost + MultiQuery (fallback)
# - Sanity check dell’indice per chunk “vuoti”
# - Nessuna reindicizzazione in runtime

from __future__ import annotations
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # disattiva telemetria Chroma

from pathlib import Path
from typing import List

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# ============================
# Config base pagina
# ============================
st.set_page_config(page_title="PDF Chat • Chroma", page_icon="📄", layout="wide")

# ----------------------------
# Accesso opzionale con password
# ----------------------------
if "APP_PASSWORD" in st.secrets:
    st.session_state._auth = st.session_state.get("_auth", False)
    if not st.session_state._auth:
        with st.sidebar:
            st.markdown("### 🔐 Accesso")
            pwd = st.text_input("Password", type="password")
            if st.button("Entra"):
                if pwd == st.secrets["APP_PASSWORD"]:
                    st.session_state._auth = True
                else:
                    st.error("Password errata.")
        if not st.session_state._auth:
            st.stop()

# ----------------------------
# Chiave OpenAI (Secrets o ENV)
# ----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY non trovato nei Secrets o nelle variabili d'ambiente.")

# ============================
# UI e impostazioni
# ============================
DEFAULT_CHROMA_DIR = "chroma_db_text_embedding_3_large_single"  # <-- nome cartella nel repo

with st.sidebar:
    st.header("⚙️ Impostazioni")
    persist_dir_input = st.text_input("Cartella Chroma (persist_directory)", value=DEFAULT_CHROMA_DIR)
    k = st.slider("Passaggi dal retriever (k)", min_value=5, max_value=20, value=12)
    temperature = st.slider("Creatività (temperature)", 0.0, 1.0, 0.0, 0.1)

st.title("📄 Chat con PDF — indice Chroma esistente")
st.caption("Questa app **non** ricostruisce l’indice. Usa una cartella Chroma già pronta e coerente con il modello di embedding.")

# ============================
# Prompt RAG
# ============================
SYSTEM_PROMPT = (
    "Sei un assistente che risponde in modo conciso e accurato usando SOLO il contenuto "
    "dei documenti forniti. Se l'informazione non è nei documenti, dillo esplicitamente. "
    "Riporta sempre riferimenti a pagina/source."
)
PROMPT_TMPL = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Domanda: {question}\n\nContesto (estratti):\n{context}\n\nRispondi in italiano.")
])

@st.cache_resource(show_spinner=False)
def get_llm(temperature: float):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=temperature)

# ============================
# Utilità
# ============================
def format_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "sorgente"
        page = meta.get("page") or meta.get("page_number")
        tag = f"[{i}] {src}" + (f", p.{page}" if page is not None else "")
        blocks.append(f"{tag}:\n{d.page_content}")
    return "\n\n".join(blocks)

@st.cache_resource(show_spinner=False)
def load_vectorstore(persist_directory: str) -> Chroma:
    """Apre la persistenza Chroma esistente in sola lettura."""
    base_dir = Path(__file__).parent.resolve()
    persist_path = (base_dir / persist_directory).resolve()

    if not persist_path.exists():
        raise RuntimeError(
            f"La cartella Chroma non esiste: '{persist_path}'. "
            "Verifica che sia presente nel repository (o aggiorna il nome in sidebar)."
        )

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")
    try:
        vs = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )
        # sanity check rapido
        _ = vs.similarity_search("test", k=1)
    except Exception as e:
        raise RuntimeError(f"Errore nell'aprire la persistenza Chroma in '{persist_path}': {e}")
    return vs

def _avg_len(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(t or "") for t in texts) / len(texts)

@st.cache_resource(show_spinner=False)
def index_health_check(_retriever) -> float:
    """Valuta media caratteri nei chunk recuperati da una probe query."""
    try:
        # prova a chiedere più risultati
        docs = _retriever.get_relevant_documents("probe query")
        return _avg_len([d.page_content for d in docs])
    except Exception:
        return 0.0

def keyword_tokens(q: str) -> list[str]:
    q = (q or "").lower()
    base = [t.strip() for t in q.replace(",", " ").split() if t.strip()]
    extra = []
    if "script" in q:
        extra += ["scripts", "script crm", "script di chiamata", "script telefonico"]
    return list(dict.fromkeys(base + extra))

def keyword_boost(query: str, docs: List[Document], top: int = 8) -> List[Document]:
    toks = keyword_tokens(query)
    if not toks:
        return docs
    scored = []
    for d in docs:
        txt = (d.page_content or "").lower()
        hit = any(t in txt for t in toks)
        scored.append((1 if hit else 0, len(txt), d))
    # prima chi ha match keyword, poi quelli con più testo
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ordered = [d for _, _, d in scored]
    # porta in alto i migliori 'top', poi il resto
    head = ordered[:top]
    tail = [d for d in ordered if d not in head]
    return head + tail

def retrieve_with_boost(query: str, base_retriever, k_expand: int = 40) -> List[Document]:
    # 1) prima passata (MMR già configurato su retriever)
    try:
        docs = base_retriever.get_relevant_documents(query)
    except AttributeError:
        docs = base_retriever.invoke(query)

    docs = keyword_boost(query, docs, top=12)

    have_keyword = any(
        any(t in (d.page_content or "").lower() for t in keyword_tokens(query)) for d in docs
    )
    poor_text = _avg_len([d.page_content for d in docs]) < 80

    # 2) se poveri o senza match lessicale -> MultiQuery
    if not have_keyword or poor_text:
        try:
            mq = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.0)
            )
            more = mq.get_relevant_documents(query)
            # unisci e deduplica
            seen = set()
            merged = []
            for d in (docs + more):
                key = (d.metadata.get("source"), d.metadata.get("page"), (d.page_content or "")[:80])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)
            docs = keyword_boost(query, merged[:k_expand], top=12)
        except Exception:
            pass

    # 3) fallback “deep” se ancora poveri
    if _avg_len([d.page_content for d in docs]) < 80:
        try:
            deep = base_retriever.vectorstore.similarity_search(query, k=max(k_expand, 50))
            docs = keyword_boost(query, deep, top=12)
        except Exception:
            pass
    return docs

# ============================
# Caricamento Vectorstore & Retriever
# ============================
with st.spinner("Carico il DB Chroma esistente…"):
    try:
        vectorstore = load_vectorstore(persist_dir_input)
    except Exception as e:
        st.error(str(e))
        st.stop()

# retriever MMR (diversificazione risultati)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": max(12, k), "lambda_mult": 0.2},
)

# Health check indice
avg_chars = index_health_check(retriever)
if avg_chars < 80:
    st.warning(
        "⚠️ L’indice Chroma sembra contenere chunk con poco testo (media < 80 caratteri). "
        "Se il PDF è scanner/immagine o l’estrazione è fallita, valuta **re-indicizzazione con OCR**. "
        "In ogni caso ho attivato query expander + keyword boost per aumentare il recall."
    )

# ============================
# Chat UI
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []

user_q = st.chat_input("Fai una domanda sul PDF indicizzato…")

# cronologia visiva
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Recupero contesto…"):
            docs = retrieve_with_boost(user_q, retriever, k_expand=40)

        if not docs:
            st.write("Non ho trovato passaggi pertinenti nell'indice.")
            st.stop()

        context = format_docs(docs)
        llm = get_llm(temperature)
        msg = PROMPT_TMPL.format_messages(question=user_q, context=context)

        with st.spinner("Genero la risposta…"):
            resp = llm.invoke(msg)
        answer = resp.content
        st.markdown(answer)

        with st.expander("Mostra estratti e citazioni"):
            for i, d in enumerate(docs, start=1):
                md = d.metadata or {}
                src = md.get("source") or md.get("file_path") or "sorgente"
                page = md.get("page") or md.get("page_number")
                label = f"[{i}] {src}" + (f" — p.{page}" if page is not None else "")
                preview = (d.page_content or "").strip()
                if not preview:
                    preview = "_(chunk vuoto)_"
                st.markdown(f"**{label}**\n\n{preview}\n")

        st.session_state.messages.append(("assistant", answer))

st.caption(
    "🔎 Retrieval via Chroma (persist_directory). Nessuna reindicizzazione in runtime. "
    "Assicurati che la cartella contenga l’indice costruito con lo stesso modello di embedding."
)

# streamlit_app.py
# Chat su indice Chroma già esistente (nessuna re-indicizzazione a runtime)

from __future__ import annotations
import os
from pathlib import Path
from typing import List

# disattiva telemetria Chroma in Cloud
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ------------------------- Config pagina -------------------------
st.set_page_config(page_title="PDF Chat • Chroma", page_icon="📄", layout="wide")

# ------------------------- Gate password (opzionale) -------------------------
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

# ------------------------- OpenAI key -------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY non trovata nei Secrets o nelle variabili d'ambiente.")

# ------------------------- UI base -------------------------
st.title("📄 Chat con PDF — indice Chroma esistente")
st.caption("Questa app **non** ricostruisce l'indice. Usa una cartella Chroma già pronta.")

DEFAULT_CHROMA_DIR = "chroma_db_v2_text_embedding_3_large_single"  # nome cartella nel repo

with st.sidebar:
    st.header("⚙️ Impostazioni")
    persist_dir_input = st.text_input("Cartella Chroma (persist_directory)", value=DEFAULT_CHROMA_DIR)
    k = st.slider("Passaggi dal retriever (k)", min_value=2, max_value=10, value=5)
    temperature = st.slider("Creatività (temperature)", 0.0, 1.0, 0.0, 0.1)

# ------------------------- Prompt RAG -------------------------
SYSTEM_PROMPT = (
    "Sei un assistente che risponde in modo conciso e accurato usando SOLO il contenuto "
    "dei documenti forniti. Se l'informazione non è nei documenti, dillo esplicitamente. "
    "Riporta sempre riferimenti a pagina/source."
)
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Domanda: {question}\n\nContesto (estratti):\n{context}\n\nRispondi in italiano."),
])

@st.cache_resource(show_spinner=False)
def get_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=temperature)

# ------------------------- Utility -------------------------
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
    """
    Apre una persistenza Chroma già esistente, senza scrivere nulla.
    Path risolto relativo al file dell'app (funziona su Streamlit Cloud).
    """
    base_dir = Path(__file__).parent.resolve()
    persist_path = (base_dir / persist_directory).resolve()

    if not persist_path.exists():
        raise RuntimeError(
            f"La cartella Chroma non esiste: '{persist_path}'. "
            "Verifica che sia presente nel repository (o aggiorna il nome in sidebar)."
        )

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")

    client_settings = ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=str(persist_path),
    )

    # solo apertura in lettura; se il DB non è compatibile, il sanity-check fallirà
    vs = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    # sanity check minimale
    _ = vs.similarity_search("test", k=1)
    return vs

# ------------------------- Carica DB e costruisci retriever -------------------------
with st.spinner("Carico il DB Chroma esistente…"):
    try:
        vectorstore = load_vectorstore(persist_dir_input)
    except Exception as e:
        st.error(str(e))
        st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": k})

# ------------------------- Chat UI -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_q = st.chat_input("Fai una domanda sul PDF indicizzato…")

# Storia a schermo
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        # 1) Recupero contesto
        with st.spinner("Recupero contesto…"):
            docs = retriever.get_relevant_documents(user_q)
        context = format_docs(docs)

        # 2) Chiamata LLM
        llm = get_llm(temperature)
        msg = PROMPT_TEMPLATE.format_messages(question=user_q, context=context)
        with st.spinner("Genero la risposta…"):
            resp = llm.invoke(msg)
        answer = resp.content

        # 3) Output
        st.markdown(answer)

        # 4) Citazioni / estratti
        with st.expander("Mostra estratti e citazioni"):
            for i, d in enumerate(docs, start=1):
                meta = d.metadata or {}
                src = meta.get("source") or meta.get("file_path") or "sorgente"
                page = meta.get("page") or meta.get("page_number")
                label = f"[{i}] {src}" + (f" — p.{page}" if page is not None else "")
                st.markdown(f"**{label}**")
                st.write(d.page_content)
                st.write("—")

        # aggiorna history
        st.session_state.messages.append(("assistant", answer))

# ------------------------- Footer -------------------------
st.caption(
    "🔎 Retrieval via Chroma (persist_directory). Nessuna reindicizzazione in runtime. "
    "Assicurati che la cartella contenga l'indice costruito con lo stesso modello di embedding."
)

import os
import streamlit as st
from typing import List

# --- Basic page config ---
st.set_page_config(page_title="PDF Chat • Chroma", page_icon="📄", layout="wide")

# --- Optional password gate ---
pwd_ok = True
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

# --- Imports for LangChain / OpenAI / Chroma ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- Settings (ENV or Secrets) ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.warning("⚠️ OPENAI_API_KEY non trovato nei secrets o nelle variabili d'ambiente.")

# Path della directory Chroma già indicizzata (persistenza)
DEFAULT_CHROMA_DIR = "chroma_db_text_embedding_3_large_single"  # <— aggiorna se il nome cartella è diverso

with st.sidebar:
    st.header("⚙️ Impostazioni")
    persist_dir = st.text_input("Cartella Chroma (persist_directory)", value=DEFAULT_CHROMA_DIR)
    k = st.slider("Numero di passaggi dal retriever (k)", min_value=2, max_value=10, value=5)
    temperature = st.slider("Creatività (temperature)", 0.0, 1.0, 0.0, 0.1)

st.title("📄 Chat con PDF — indice Chroma esistente")
st.caption("Questa app **non** ricostruisce l'indice. Usa una cartella Chroma già pronta.")

# --- Lazy init dell'Embedding e del Vectorstore esistente ---
@st.cache_resource(show_spinner=False)
def load_vectorstore(persist_directory: str):
    # Non costruisce nulla: *solo* apre la persistenza Chroma esistente
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")
    try:
        vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    except Exception as e:
        raise RuntimeError(f"Errore nell'aprire la cartella Chroma '{persist_directory}': {e}")
    # Sanity check minimo: conta gli items
    try:
        # Chroma non espone direttamente il count; usiamo una query a-vuoto per verificare.
        _ = vs.similarity_search("test", k=1)
    except Exception as e:
        raise RuntimeError(f"La directory esiste ma non sembra un DB Chroma valido: {e}")
    return vs

# --- Prompt base per il RAG ---
SYSTEM_PROMPT = (
    "Sei un assistente che risponde in modo conciso e accurato usando SOLO il contenuto dei documenti forniti. "
    "Se la risposta non è nei documenti, dì che non è presente. Riporta sempre riferimenti a pagina/source."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Domanda: {question}\n\nContesto (estratti):\n{context}\n\nRispondi in italiano."),
])

# --- LLM ---
@st.cache_resource(show_spinner=False)
def get_llm(temperature: float):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=temperature)

# --- Utility per formattare contesto e citazioni ---
def format_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "sorgente"
        page = meta.get("page") or meta.get("page_number")
        tag = f"[{i}] {src}"
        if page is not None:
            tag += f", p.{page}"
        blocks.append(f"{tag}:\n{d.page_content}")
    return "\n\n".join(blocks)

# --- Caricamento del vectorstore ---
with st.spinner("Carico il DB Chroma esistente…"):
    try:
        vectorstore = load_vectorstore(persist_dir)
    except Exception as e:
        st.error(str(e))
        st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": k})

# --- UI Chat ---
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
        with st.spinner("Recupero contesto…"):
            docs = retriever.get_relevant_documents(user_q)
        context = format_docs(docs)

        llm = get_llm(temperature)
        msg = PROMPT_TEMPLATE.format_messages(question=user_q, context=context)
        with st.spinner("Genero la risposta…"):
            resp = llm.invoke(msg)
        answer = resp.content

        # Mostra risposta
        st.markdown(answer)

        # Pannello citazioni/estratti
        with st.expander("Mostra estratti e citazioni"):
            for i, d in enumerate(docs, start=1):
                meta = d.metadata or {}
                src = meta.get("source") or meta.get("file_path") or "sorgente"
                page = meta.get("page") or meta.get("page_number")
                label = f"[{i}] {src}"
                if page is not None:
                    label += f" — p.{page}"
                st.markdown(f"**{label}**")
                st.write(d.page_content)
                st.write("—")

        # Log in history
        st.session_state.messages.append(("assistant", answer))

# --- Footer ---
st.caption(
    "🔎 Retrieval via Chroma (persist_directory). Nessuna reindicizzazione in runtime. "
    "Assicurati che la cartella contenga l'indice costruito con lo stesso modello di embedding."
)

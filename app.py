import os
import shutil
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        from langchain.llms import Ollama
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3

PERSIST_DIR = os.path.join(tempfile.gettempdir(), "rag_local_chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL_NAME = "mistral"

RAG_PROMPT_TEMPLATE = """Tu es mon assistant documentaire. Voici comment tu dois faire:

- Tu ne réponds qu'avec ce qu'il y a dans le texte ci-dessous. Tu ne dois pas inventer de réponse.
- Si l'information que l'utilisateur cherche n'est pas dans le document, dis-le directement que tu n'as pas trouvé de réponse... n'essaie pas de deviner.
- Garde la même langue que celle utilisée dans la question.

Texte de référence :
{context}

Question posée : {question}

Ta réponse :"""

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm():
    return Ollama(model=OLLAMA_MODEL_NAME, temperature=0)


def load_document(file_path: str, file_name: str):
    extension = os.path.splitext(file_name)[1].lower()

    if extension == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif extension in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Extension non supportée : {extension}")

    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = file_name

    return documents


def ingest_files(uploaded_files, embedding_model) -> Chroma:
    """
    Étape 2 — Pipeline d'ingestion complet :
        Extraction -> Chunking -> Vectorisation -> Stockage dans Chroma.
    """
    all_documents = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_document(tmp_path, uploaded_file.name)
            all_documents.extend(docs)

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_documents)

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
    )
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    return vectorstore, len(all_documents), len(chunks)


def semantic_search(vectorstore: Chroma, query: str):
    results = vectorstore.similarity_search(query, k=TOP_K)
    return results


def rag_answer(vectorstore: Chroma, llm, query: str):
    """
    Alainy ny contexte pertinent, amboariny ny prompt via PromptTemplate,
    de avy eo anontaniny le LLM local. Retourne la reponse et les chunks sources utilises.
    """
    source_chunks = semantic_search(vectorstore, query)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in source_chunks)

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    final_prompt = prompt.format(context=context_text, question=query)

    answer = llm.invoke(final_prompt)
    return answer, source_chunks


#  Interface Streamlit
st.set_page_config(page_title="RAG Local — Clone NotebookLM", page_icon="📚", layout="wide")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # liste de dicts {role, content, sources?}
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
with st.sidebar:
    st.title("RAG Local")
    st.caption("Aucune donnée ne quitte votre machine.")
    st.subheader("1. Charger des documents")
    uploaded_files = st.file_uploader(
        "PDF, Markdown ou TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    index_button = st.button("Indexer les documents", use_container_width=True)
    if index_button:
        if not uploaded_files:
            st.warning("Veuillez charger au moins un fichier avant d'indexer.")
        else:
            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                embedding_model = get_embedding_model()
                vectorstore, nb_docs, nb_chunks = ingest_files(uploaded_files, embedding_model)
                st.session_state.vectorstore = vectorstore
                st.session_state.indexed_files = [f.name for f in uploaded_files]
            st.success(f"{nb_docs} document(s) indexé(s) en {nb_chunks} chunks.")
    if st.session_state.indexed_files:
        st.caption("Fichiers indexés :")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")
    st.divider()
    st.subheader("2. Mode de fonctionnement")
    llm_enabled = st.toggle(
        "Activer l'assistant LLM (mode RAG complet)",
        value=False,
        help="Désactivé : recherche sémantique pure (extraits bruts). "
             "Activé : génération de réponse par le LLM local à partir du contexte.",
    )
    st.caption(
        "Recherche sémantique pure" if not llm_enabled else "Assistant RAG complet"
    )
st.header("Interroger vos documents")
if st.session_state.vectorstore is None:
    st.info("Chargez et indexez des documents dans la barre latérale pour commencer.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Extraits sources utilisés"):
                for i, doc in enumerate(message["sources"], start=1):
                    st.markdown(f"**Extrait {i} — source : `{doc.metadata.get('source', 'inconnu')}`**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
user_query = st.chat_input("Posez une question sur vos documents...")
if user_query:
    if st.session_state.vectorstore is None:
        st.error("Veuillez d'abord indexer des documents.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            if not llm_enabled:
                with st.spinner("Recherche dans la base vectorielle..."):
                    results = semantic_search(st.session_state.vectorstore, user_query)
                if not results:
                    response_text = "Aucun extrait pertinent trouvé dans les documents indexés."
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    response_text = f"Voici les {len(results)} extraits les plus proches de votre requête :"
                    st.markdown(response_text)
                    for i, doc in enumerate(results, start=1):
                        st.markdown(f"**Extrait {i} — source : `{doc.metadata.get('source', 'inconnu')}`**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text, "sources": results}
                    )
            else:
                with st.spinner("Génération de la réponse par le modèle local..."):
                    llm = get_llm()
                    try:
                        answer, sources = rag_answer(st.session_state.vectorstore, llm, user_query)
                    except Exception as e:
                        answer = (
                            "Impossible de contacter le modèle Ollama local. "
                            f"Vérifiez qu'Ollama tourne bien (`ollama run {OLLAMA_MODEL_NAME}`).\n\n"
                            f"Détail technique : {e}"
                        )
                        sources = []
                st.markdown(answer)
                if sources:
                    with st.expander("📎 Extraits sources utilisés pour cette réponse"):
                        for i, doc in enumerate(sources, start=1):
                            st.markdown(f"**Extrait {i} — source : `{doc.metadata.get('source', 'inconnu')}`**")
                            st.markdown(doc.page_content)
                            st.markdown("---")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

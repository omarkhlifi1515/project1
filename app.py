import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from rag_system import RAGSystem, ModelConfig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ENISO Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark gradient header */
    .stApp > header { background: transparent; }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Category buttons */
    .category-btn {
        display: inline-block;
        padding: 6px 14px;
        margin: 3px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        font-size: 0.85rem;
        cursor: pointer;
    }
    
    /* Source pills */
    .source-pill {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
        font-size: 0.78rem;
    }

    /* Gold source pill */
    .source-pill.gold {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Feedback buttons */
    .feedback-row {
        display: flex;
        gap: 8px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached RAG initialization (persists across reruns & sessions)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_rag_system(model_preset: str) -> RAGSystem:
    """Initialize the RAG system once and cache it across sessions.
    The LLM + Vector DB connection is created here.
    Subsequent calls with the same model_preset return the cached instance."""
    return RAGSystem(model_preset=model_preset)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    available_models = list(ModelConfig.PRESETS.keys())
    selected_model = st.selectbox(
        "Modèle LLM",
        available_models,
        index=0,
        help="Choisir le modèle Groq à utiliser",
    )

    st.markdown("---")
    st.markdown("### 🏷️ Catégories rapides")
    st.caption("Cliquez pour pré-remplir une question type")

    quick_questions = {
        "📅 Emploi du Temps": "Quel est l'emploi du temps du groupe G1 ?",
        "📝 Examens": "Quelles sont les dates des examens DS ?",
        "🏢 Stages": "Comment faire un stage d'été ? Quelles sont les démarches ?",
        "🎓 PFE": "Quels sont les sujets PFE disponibles ?",
        "📋 Inscription": "Quelles sont les procédures d'inscription ?",
    }

    for label, question in quick_questions.items():
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state["prefill_question"] = question

    st.markdown("---")

    # Update / Sync button
    if st.button("🔄 Mettre à jour / Reprendre l'index", use_container_width=True):
        with st.spinner("♻️ Synchronisation des documents (peut prendre 1 min si limite de quota atteinte)..."):
            rag = init_rag_system(selected_model)
            rag.sync_index()
            st.success("✅ Index mis à jour !")

    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state["messages"] = []
        rag = init_rag_system(selected_model)
        rag.clear_history()
        st.rerun()

    st.markdown("---")

    # Gold standard stats
    import os
    gold_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_standard")
    gold_count = len(list(Path(gold_dir).glob("*.md"))) if os.path.exists(gold_dir) else 0
    st.markdown(f"⭐ **Réponses validées:** {gold_count}")

    st.markdown("---")
    st.markdown(
        "<small style='color: #666'>Powered by Groq + HuggingFace + LangChain<br>"
        "Data: ENISO Documents</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
from pathlib import Path

st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>🎓 ENISO Assistant</h1>
    <p style='text-align: center; color: #888; margin-top: 4px;'>
        Assistant IA basé sur les documents officiels de l'ENISO — Propulsé par Groq & HuggingFace
    </p>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages with feedback buttons
for idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    pill_class = "source-pill gold" if "⭐" in src else "source-pill"
                    st.markdown(f'<span class="{pill_class}">{src}</span>', unsafe_allow_html=True)

        # Feedback buttons for assistant messages
        if msg["role"] == "assistant" and not msg.get("feedback"):
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{idx}", help="Cette réponse est bonne"):
                    # Find the user question that preceded this answer
                    user_question = ""
                    for j in range(idx - 1, -1, -1):
                        if st.session_state["messages"][j]["role"] == "user":
                            user_question = st.session_state["messages"][j]["content"]
                            break

                    if user_question:
                        rag = init_rag_system(selected_model)
                        rag.save_gold_standard(user_question, msg["content"])
                        st.session_state["messages"][idx]["feedback"] = "up"
                        st.toast("⭐ Réponse sauvegardée comme référence !", icon="✅")
                        st.rerun()
            with col2:
                if st.button("👎", key=f"down_{idx}", help="Cette réponse est mauvaise"):
                    st.session_state["messages"][idx]["feedback"] = "down"
                    st.toast("📝 Merci pour votre retour !", icon="🔄")
                    st.rerun()

        # Show feedback state
        elif msg["role"] == "assistant" and msg.get("feedback") == "up":
            st.markdown("✅ *Réponse validée comme référence*")
        elif msg["role"] == "assistant" and msg.get("feedback") == "down":
            st.markdown("🔄 *Merci pour votre retour*")

# Handle prefilled question from sidebar
prefill = st.session_state.pop("prefill_question", None)

# Chat input
user_input = st.chat_input("Posez votre question sur l'ENISO...")

# Use prefill if no direct input
if prefill and not user_input:
    user_input = prefill

if user_input:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate response with streaming
    rag = init_rag_system(selected_model)

    with st.chat_message("assistant", avatar="🎓"):
        # Step 1: Retrieve context (fast)
        with st.spinner("🔍 Recherche dans les documents ENISO..."):
            prompt, context_results = rag._build_prompt(user_input)

        # Step 2: Stream LLM response
        try:
            def token_generator():
                for chunk in rag.model.stream(prompt):
                    # ChatOpenAI returns AIMessageChunk objects
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        yield text

            full_response = st.write_stream(token_generator())

            # Ensure full_response is a string
            if not isinstance(full_response, str):
                full_response = str(full_response) if full_response else ""

        except Exception as e:
            # Fallback: non-streaming mode
            st.warning(f"Streaming failed, using standard mode: {e}")
            response = rag.model.invoke(prompt)
            full_response = response.content if hasattr(response, "content") else str(response)
            st.markdown(full_response)

        # Finalize (update history, get sources)
        sources = rag._finalize_response(user_input, full_response, context_results)

        if sources:
            with st.expander("📄 Sources", expanded=False):
                for src in sources:
                    pill_class = "source-pill gold" if "⭐" in src else "source-pill"
                    st.markdown(f'<span class="{pill_class}">{src}</span>', unsafe_allow_html=True)

    # Save assistant message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

    st.rerun()  # Rerun to show feedback buttons on the new message

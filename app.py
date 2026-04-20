import streamlit as st
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
</style>
""", unsafe_allow_html=True)


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
        help="Choisir le modèle Ollama à utiliser",
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
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state["messages"] = []
        if "rag_system" in st.session_state:
            st.session_state["rag_system"].clear_history()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color: #666'>Powered by Ollama + LangChain<br>"
        "Data: ENISO Documents</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load or reload RAG system
# ---------------------------------------------------------------------------
def get_rag_system(model_preset: str) -> RAGSystem:
    """Get or create RAG system, respecting model changes."""
    if (
        "rag_system" not in st.session_state
        or st.session_state.get("current_model") != model_preset
    ):
        with st.spinner(f"🔄 Chargement du système RAG (modèle: {model_preset})..."):
            st.session_state["rag_system"] = RAGSystem(model_preset=model_preset)
            st.session_state["current_model"] = model_preset
    return st.session_state["rag_system"]


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>🎓 ENISO Assistant</h1>
    <p style='text-align: center; color: #888; margin-top: 4px;'>
        Assistant IA basé sur les documents officiels de l'ENISO
    </p>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-pill">{src}</span>', unsafe_allow_html=True)

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
    rag = get_rag_system(selected_model)

    with st.chat_message("assistant", avatar="🎓"):
        # Step 1: Retrieve context (fast)
        with st.spinner("🔍 Recherche dans les documents ENISO..."):
            prompt, context_results = rag._build_prompt(user_input)

        # Step 2: Stream LLM response
        try:
            def token_generator():
                for chunk in rag.model.stream(prompt):
                    # Ensure chunk is a string
                    text = str(chunk) if not isinstance(chunk, str) else chunk
                    if text:
                        yield text

            full_response = st.write_stream(token_generator())

            # Ensure full_response is a string
            if not isinstance(full_response, str):
                full_response = str(full_response) if full_response else ""

        except Exception as e:
            # Fallback: non-streaming mode
            st.warning(f"Streaming failed, using standard mode...")
            full_response = rag.model.invoke(prompt)
            st.markdown(full_response)

        # Finalize (update history, get sources)
        sources = rag._finalize_response(user_input, full_response, context_results)

        if sources:
            with st.expander("📄 Sources", expanded=False):
                for src in sources:
                    st.markdown(f'<span class="source-pill">{src}</span>', unsafe_allow_html=True)

    # Save assistant message
    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

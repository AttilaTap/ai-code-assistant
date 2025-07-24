import streamlit as st
from run import load_codebase_and_start_chat, create_snippet_chain, get_available_ollama_models

st.set_page_config(page_title="Codebase Chat", layout="wide")
st.title("💬 AI Codebase Assistant")

# --- MODE SELECTOR ---
mode = st.radio("Choose mode:", ["Chat with a Codebase", "Paste Code Snippet"], horizontal=True)

# --- INIT STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- MODEL DROPDOWN ---
available_models = get_available_ollama_models()
preferred_default = "codellama:13b-instruct"

if not available_models:
    st.warning("⚠️ No Ollama models found. Use `ollama pull` to download a model.")
    selected_model = None
else:
    # Ensure preferred model is at the top
    if preferred_default in available_models:
        available_models.remove(preferred_default)
        available_models.insert(0, preferred_default)

    selected_model = st.selectbox("🤖 Choose an LLM model", available_models, index=0)

# --- CODEBASE MODE ---
if mode == "Chat with a Codebase":
    col1, col2 = st.columns(2)
    with col1:
        repo_url = st.text_input("🔗 GitHub repository URL", placeholder="https://github.com/you/project")
    with col2:
        local_path = st.text_input("📁 Or local code folder path", placeholder="/path/to/code")

    if st.button("🚀 Load Codebase"):
        with st.spinner("Processing..."):
            try:
                if repo_url:
                    st.session_state.qa_chain = load_codebase_and_start_chat(git_url=repo_url, model_name=selected_model)
                elif local_path:
                    st.session_state.qa_chain = load_codebase_and_start_chat(code_path=local_path, model_name=selected_model)
                else:
                    st.error("❗ Please provide a GitHub URL or local path.")
            except Exception as e:
                st.error(f"❌ Failed to load codebase: {e}")

# --- SNIPPET MODE ---
elif mode == "Paste Code Snippet":
    snippet = st.text_area("Paste your code here:", height=300)
    if snippet and st.button("💬 Start Chat About Snippet"):
        try:
            st.session_state.qa_chain = create_snippet_chain(snippet, model_name=selected_model)
            st.success("Snippet loaded. Ask your question below.")
        except Exception as e:
            st.error(f"❌ Failed to process snippet: {e}")

# --- CHAT SECTION (Common) ---
if st.session_state.qa_chain:
    st.subheader("🧠 Ask a Question")
    user_input = st.text_input("Your question", placeholder="e.g. What does this function do?")
    if user_input:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": user_input})
                st.session_state.chat_history.append((user_input, result["result"]))
            except Exception as e:
                st.session_state.chat_history.append((user_input, f"⚠️ Error: {e}"))

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🧠 You:** {q}")
        st.markdown(f"**🤖 LLM:** {a}")
        st.markdown("---")

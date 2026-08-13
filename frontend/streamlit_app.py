import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 Documents")
    st.caption("Manage your knowledge base")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.button("📥 Index Document", use_container_width=True):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            with st.spinner("Indexing document..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                )

            if response.status_code == 200:
                data = response.json()

                st.session_state.indexed_docs.append(
                    {
                        "name": data["filename"],
                        "chunks": data["chunks_created"],
                    }
                )

                st.success("Document indexed successfully!")
            else:
                st.error(response.text)

    st.divider()

    st.subheader("Indexed Documents")

    if st.session_state.indexed_docs:
        for doc in st.session_state.indexed_docs:
            st.markdown(
                f"**{doc['name']}**  \n{doc['chunks']} chunks indexed"
            )
    else:
        st.info("No documents indexed in this session.")

    st.divider()

    st.subheader("System Status")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2)

        if health.status_code == 200:
            st.success("Backend connected")
        else:
            st.error("Backend error")
    except Exception:
        st.error("Backend offline")

    st.divider()

    st.subheader("Conversation")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Main Chat Area ----------
st.title("RAG PDF Assistant")
st.caption(
    "Ask questions about your uploaded documents and receive grounded answers with citations."
)

# Display previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.markdown("**Sources**")

            for source in message["sources"]:
                st.markdown(
                    f"> 📄 **{source.get('source', 'Unknown')}**  \n> Page {source.get('page', '?')}"
                )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("_Thinking..._")

        start_time = time.time()

        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"question": prompt},
                timeout=120,
            )

            latency = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer returned.",
                )

                sources = result.get("sources", [])

                typing_placeholder.empty()

                stream_placeholder = st.empty()
                streamed = ""

                for word in answer.split():
                    streamed += word + " "
                    stream_placeholder.markdown(streamed)
                    time.sleep(0.02)

                if sources:
                    st.markdown("**Sources**")

                    for source in sources:
                        st.markdown(
                            f"> 📄 **{source.get('source', 'Unknown')}**  \n> Page {source.get('page', '?')}"
                        )

                with st.expander("Developer metrics"):
                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Response time",
                        f"{latency:.2f}s",
                    )

                    col2.metric(
                        "Retrieved sources",
                        len(sources),
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            else:
                typing_placeholder.empty()
                st.error(
                    f"Backend error ({response.status_code}): {response.text}"
                )

        except requests.exceptions.RequestException as e:
            typing_placeholder.empty()
            st.error(f"Connection error: {e}")
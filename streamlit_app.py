from uuid import uuid4

import streamlit as st

from src.rag_study_buddy.rag import RagSession, SOURCE_PAGES, describe_error, load_settings

def render_evidence(evidence):
    if not evidence:
        st.info("Supporting passages will appear here after you ask a question.")
        return
    st.caption(f"{len(evidence)} retrieved passages · Check these against the answer.")
    for passage in evidence:
        st.markdown(f'<div id="{passage.anchor}"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            name = next((name for name, url in SOURCE_PAGES.items() if url == passage.source), "Document")
            st.markdown(f"**[{passage.number}] {name}**")
            st.caption(f"Chunk {passage.chunk_id}")
            st.text(passage.text)
            st.link_button("Open source ↗", passage.source, width="stretch")


def clear_conversation():
    st.session_state.turns = []
    st.session_state.pop("evidence_turn", None)
    st.session_state.pop("pending_evidence_turn", None)


def select_evidence(turn_id):
    st.session_state.evidence_turn = turn_id


def render_citation_controls(turn):
    if not turn["evidence"]:
        return
    with st.container(horizontal=True):
        st.button(
            "View evidence", key=f"view-{turn['id']}",
            on_click=select_evidence, args=(turn["id"],),
        )
        for passage in turn["evidence"]:
            with st.popover(f"[{passage.number}]", help="Read this supporting passage"):
                st.caption(f"Chunk {passage.chunk_id}")
                st.text(passage.text)
                st.link_button("Open source ↗", passage.source)


def main():
    st.set_page_config(page_title="RAG Study Buddy", page_icon="📖", layout="wide")
    st.markdown("""
    <style>
    .block-container { max-width: 1440px; padding-top: 4rem; }
    h1 { letter-spacing: -0.045em; }
    [data-testid="stSidebar"] { border-right: 1px solid #dde5de; }
    [data-testid="stText"] { white-space: pre-wrap; overflow-wrap: anywhere; }
    @media (max-width: 760px) {
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stColumn"] { min-width: 100% !important; }
        .block-container { padding: 4rem 1rem 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)
    if "turns" not in st.session_state:
        st.session_state.turns = []
        st.session_state.rag_session = None
        st.session_state.settings_identity = None

    with st.sidebar:
        st.markdown("### STUDY BUDDY")
        st.caption("Your documents. Answers you can check.")
        st.divider()
        st.markdown("**DOCUMENT LIBRARY**")
        selected = st.multiselect(
            "Search these documents", list(SOURCE_PAGES), default=list(SOURCE_PAGES), key="documents",
        )
        for name in selected:
            st.markdown(f"[↗ {name}]({SOURCE_PAGES[name]})")
        st.divider()
        try:
            settings = load_settings([SOURCE_PAGES[name] for name in selected])
        except Exception as error:
            st.error(describe_error(error))
            return
        st.markdown("**LOCAL MODELS**")
        st.caption(f"Chat: {settings.chat_model}")
        st.caption(f"Embeddings: {settings.embedding_model}")
        ready_to_prepare = bool(selected)
        identity = settings
        if st.session_state.settings_identity != identity:
            st.session_state.settings_identity = identity
            st.session_state.rag_session = None
            clear_conversation()
        prepare = st.button("Prepare documents", disabled=not ready_to_prepare, width="stretch", key="prepare")
        if st.session_state.rag_session:
            st.success(f"Ready · {st.session_state.rag_session.chunk_count} passages")
        st.button("New conversation", on_click=clear_conversation, width="stretch", key="clear")
        st.caption("Changing documents starts a new conversation. Your saved document index is kept.")

    st.caption("DOCUMENT CHAT / EVIDENCE INCLUDED")
    st.title("A little clarity, backed by your docs.")
    st.markdown("Ask a question. Follow the sources. Build your understanding.")
    st.divider()
    chat_column, evidence_column = st.columns([1.55, 1], gap="large")
    with chat_column:
        st.subheader("Conversation")
        st.caption("Answers use the selected documents and include links to supporting passages.")
        transcript = st.container(height=520, border=True)
        with transcript:
            if not st.session_state.turns:
                st.markdown("### What would you like to understand?")
                st.write("Start with a question about Deep Agents, subagents, or streaming.")
                st.caption('Try: “What is a subagent, and when should I use one?”')
            for turn in st.session_state.turns:
                with st.container():
                    st.markdown(f"**Question:** {turn['question']}")
                with st.container():
                    if turn.get("error"):
                        st.markdown("**Answer:**")
                        st.error(turn["error"])
                    else:
                        st.markdown(f"**Answer:** {turn['answer']}")
                    render_citation_controls(turn)
        question = st.chat_input(
            "Ask about your documents…", disabled=not ready_to_prepare,
            key="question", max_chars=8000, submit_mode="disable",
        )
        if not selected:
            st.info("Select at least one document to start.")

    with evidence_column:
        st.subheader("Evidence")
        st.caption("The passages retrieved for the selected answer.")
        turns = st.session_state.turns
        if turns:
            ids = [turn["id"] for turn in turns]
            if st.session_state.get("evidence_turn") not in ids:
                st.session_state.evidence_turn = ids[-1]
            labels = {turn["id"]: f"{i + 1}. {turn['question'][:70]}" for i, turn in enumerate(turns)}
            selected_turn = st.selectbox("Evidence for", ids, format_func=labels.get, key="evidence_turn")
            evidence = next(turn["evidence"] for turn in turns if turn["id"] == selected_turn)
        else:
            evidence = []
        evidence_box = st.empty()
        with evidence_box.container():
            render_evidence(evidence)

    if prepare or (question and question.strip()):
        with chat_column:
            with st.status("Preparing documents…", expanded=True) as status:
                try:
                    if st.session_state.rag_session is None:
                        st.session_state.rag_session = RagSession.prepare(settings, on_progress=status.write)
                    status.update(label="Documents ready", state="complete", expanded=False)
                except Exception as error:
                    status.update(label="Could not prepare documents", state="error")
                    st.error(describe_error(error))
                    return
        if not question or not question.strip():
            st.rerun()
        turn = {"id": uuid4().hex, "question": question.strip(), "answer": "", "evidence": []}
        history = []
        for previous in st.session_state.turns:
            if not previous.get("error"):
                history.extend([
                    {"role": "user", "content": previous["question"]},
                    {"role": "assistant", "content": previous["answer"]},
                ])
        with transcript:
            with st.container():
                st.markdown(f"**Question:** {turn['question']}")
            with st.container():
                answer_box = st.empty()
                with st.status("Searching documents…", expanded=False) as status:
                    try:
                        for event in st.session_state.rag_session.stream_answer(
                            turn["question"], history, turn_id=turn["id"],
                        ):
                            if event["type"] == "status":
                                status.update(label=event["text"])
                            elif event["type"] == "draft":
                                if event["text"]:
                                    answer_box.markdown(f"**Answer:** {event['text']} ▍")
                                else:
                                    answer_box.empty()
                            elif event["type"] == "evidence":
                                turn["evidence"] = event["evidence"]
                                with evidence_box.container():
                                    render_evidence(turn["evidence"])
                            elif event["type"] == "complete":
                                turn["answer"] = event["text"]
                                turn["evidence"] = event["evidence"]
                                answer_box.markdown(f"**Answer:** {turn['answer']}")
                        status.update(label="Answer ready", state="complete")
                    except Exception as error:
                        answer_box.markdown("**Answer:**")
                        turn["error"] = describe_error(error)
                        status.update(label="Could not complete the answer", state="error")
        st.session_state.turns.append(turn)
        st.session_state.pending_evidence_turn = turn["id"]
        st.rerun()


if __name__ == "__main__":
    if "pending_evidence_turn" in st.session_state:
        st.session_state.evidence_turn = st.session_state.pop("pending_evidence_turn")
    main()

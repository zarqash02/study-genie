"""Study Genie — Streamlit frontend with NotebookLM-style features."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
import streamlit as st
from streamlit.components.v1 import html as st_html
from backend.config import config
from backend.services import pdf_service, embedding_service, pinecone_service, llm_service

st.set_page_config(page_title="Study Genie", page_icon="📘", layout="wide")

@st.cache_resource
def init():
    config.validate()
    embedding_service.get_model()
    pinecone_service.get_index()
    return True

try:
    init()
except Exception as e:
    st.error(f"Startup failed: {e}")
    st.stop()

st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1300px; }
.doc-card { background:#f7f7f8; border:1px solid #e5e5e7; border-radius:8px; padding:10px 12px; margin-bottom:6px; font-size:0.9rem; }
.source-box { background:#f4f6fb; border-left:3px solid #4a6cf7; padding:8px 12px; margin:6px 0; border-radius:4px; font-size:0.85rem; }
.note-card { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:12px 14px; margin:8px 0; font-size:0.9rem; }
.concept-card { background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:12px 14px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "docs"          not in st.session_state: st.session_state.docs = []
if "chat"          not in st.session_state: st.session_state.chat = []
if "tool_output"   not in st.session_state: st.session_state.tool_output = None
if "last_uploaded" not in st.session_state: st.session_state.last_uploaded = None
if "quiz_answers"  not in st.session_state: st.session_state.quiz_answers = {}
if "card_idx"      not in st.session_state: st.session_state.card_idx = 0
if "card_flipped"  not in st.session_state: st.session_state.card_flipped = False
if "notes"         not in st.session_state: st.session_state.notes = []
if "progress"      not in st.session_state:
    st.session_state.progress = {
        "questions_asked": 0, "quizzes_taken": 0,
        "total_mcq_answered": 0, "total_mcq_correct": 0,
        "weak_topics": {}, "missed_questions": [],
    }


# ---------- Core helpers ----------
def process_upload(file_bytes, filename):
    doc_id = str(uuid.uuid4())[:8]
    records = pdf_service.process_pdf(
        file_bytes=file_bytes, doc_id=doc_id, doc_name=filename,
        chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
    )
    if not records:
        return None, "No extractable text in PDF."
    vectors = embedding_service.embed_texts([r["text"] for r in records])
    pinecone_service.upsert_chunks(records, vectors)
    return {"doc_id": doc_id, "doc_name": filename, "chunks": len(records)}, None


def process_text(text: str, name: str):
    doc_id = str(uuid.uuid4())[:8]
    chunks = pdf_service.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return None, "No text to process."
    records = [
        {"id": f"{doc_id}_{i}", "text": c,
         "metadata": {"doc_id": doc_id, "doc_name": name, "page": 1, "text": c}}
        for i, c in enumerate(chunks)
    ]
    vectors = embedding_service.embed_texts([r["text"] for r in records])
    pinecone_service.upsert_chunks(records, vectors)
    return {"doc_id": doc_id, "doc_name": name, "chunks": len(records)}, None


def do_ask(query, doc_ids):
    vec = embedding_service.embed_query(query)
    matches = pinecone_service.query(vec, top_k=config.TOP_K, doc_ids=doc_ids or None)
    if not matches:
        return {"answer": "I don't know based on the provided material.", "sources": []}
    answer = llm_service.answer_question(query, matches)
    sources = [{
        "doc_name": m["metadata"].get("doc_name"),
        "page": m["metadata"].get("page"),
        "score": round(m["score"], 3),
        "snippet": m["metadata"].get("text", "")[:220] + "…",
    } for m in matches]
    return {"answer": answer, "sources": sources}


def fetch_context(doc_ids, k=10):
    seed = embedding_service.embed_query("key concepts and main ideas overview")
    return pinecone_service.query(seed, top_k=k, doc_ids=doc_ids or None)


def add_note(content: str, label: str = "Note"):
    st.session_state.notes.append({"label": label, "content": content, "id": str(uuid.uuid4())[:6]})


# ---------- Mind map HTML ----------
def render_mind_map(concepts: list):
    if not concepts:
        return
    nodes = []
    edges = []
    nodes.append({"id": "center", "label": "Key Concepts", "color": "#4a6cf7", "font": {"color": "#fff"}, "size": 30})
    for i, c in enumerate(concepts):
        nid = f"c{i}"
        nodes.append({"id": nid, "label": c.get("term", ""), "color": "#6b4af7",
                       "font": {"color": "#fff"}, "title": c.get("definition", ""), "size": 20})
        edges.append({"from": "center", "to": nid})
        for j, rel in enumerate(c.get("related", [])[:2]):
            rid = f"r{i}_{j}"
            nodes.append({"id": rid, "label": rel, "color": "#a78bfa",
                           "font": {"color": "#fff"}, "size": 14})
            edges.append({"from": nid, "to": rid})

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = f"""
    <html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
    <style>
      body {{margin:0;background:#0e1117;}}
      #mynetwork {{width:100%;height:480px;border:1px solid #333;border-radius:8px;background:#0e1117;}}
    </style>
    </head><body>
    <div id="mynetwork"></div>
    <script>
      var nodes = new vis.DataSet({nodes_json});
      var edges = new vis.DataSet({edges_json});
      var container = document.getElementById('mynetwork');
      var data = {{nodes:nodes, edges:edges}};
      var options = {{
        nodes:{{shape:'box',borderWidth:0,shadow:true,font:{{size:13}}}},
        edges:{{color:'#555',smooth:{{type:'cubicBezier'}}}},
        physics:{{solver:'forceAtlas2Based',stabilization:{{iterations:150}}}},
        interaction:{{hover:true,tooltipDelay:100}}
      }};
      new vis.Network(container, data, options);
    </script>
    </body></html>
    """
    st_html(html, height=500, scrolling=False)


# ---------- Header ----------
st.title("📘 Study Genie")
st.caption("Your intelligent learning companion — grounded answers from your own materials.")

# ---------- Top-level tabs ----------
study_tab, notes_tab, progress_tab = st.tabs(["📚 Study", "📝 Notes", "📊 Progress"])


# ============================================================
# STUDY TAB
# ============================================================
with study_tab:
    left, right = st.columns([1, 2], gap="large")

    # ----- LEFT: Library + Tools -----
    with left:
        st.subheader("Library")
        upl_tab, paste_tab = st.tabs(["📄 Upload PDF", "✏️ Paste Text"])

        with upl_tab:
            uploaded = st.file_uploader("PDF file", type=["pdf"],
                                        label_visibility="collapsed", key="pdf_upl")
            if uploaded is not None and uploaded.name != st.session_state.last_uploaded:
                with st.spinner(f"Processing {uploaded.name}..."):
                    res, err = process_upload(uploaded.getvalue(), uploaded.name)
                if res:
                    st.session_state.docs.append(res)
                    st.session_state.last_uploaded = uploaded.name
                    st.success(f"Added **{res['doc_name']}** ({res['chunks']} chunks)")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {err}")
                    st.session_state.last_uploaded = uploaded.name

        with paste_tab:
            doc_name = st.text_input("Document name", placeholder="e.g. Chapter 3 notes", key="paste_name")
            pasted = st.text_area("Paste study text here", height=180, key="paste_text")
            if st.button("Add text", use_container_width=True, disabled=not (doc_name and pasted)):
                with st.spinner("Processing..."):
                    res, err = process_text(pasted, doc_name)
                if res:
                    st.session_state.docs.append(res)
                    st.success(f"Added **{res['doc_name']}** ({res['chunks']} chunks)")
                    st.rerun()
                else:
                    st.error(f"Failed: {err}")

        if st.session_state.docs:
            st.markdown("##### Documents")
            for d in st.session_state.docs:
                ca, cb = st.columns([5, 1])
                with ca:
                    st.markdown(
                        f"<div class='doc-card'>📄 {d['doc_name']}<br>"
                        f"<span style='color:#888;font-size:0.8rem;'>{d['chunks']} chunks · {d['doc_id']}</span></div>",
                        unsafe_allow_html=True)
                with cb:
                    if st.button("✕", key=f"del_{d['doc_id']}"):
                        pinecone_service.delete_doc(d["doc_id"])
                        st.session_state.docs = [x for x in st.session_state.docs if x["doc_id"] != d["doc_id"]]
                        st.rerun()
        else:
            st.info("Upload a PDF or paste text to get started.")

        st.divider()
        st.subheader("Tools")
        disabled = len(st.session_state.docs) == 0
        doc_ids = [d["doc_id"] for d in st.session_state.docs]

        if st.button("📝 Summarize", use_container_width=True, disabled=disabled):
            with st.spinner("Summarizing..."):
                chunks = fetch_context(doc_ids)
                result = llm_service.summarize(chunks) if chunks else "No content."
                st.session_state.tool_output = ("summary", result)

        if st.button("🧠 Key Concepts", use_container_width=True, disabled=disabled):
            with st.spinner("Extracting concepts..."):
                chunks = fetch_context(doc_ids)
                result = llm_service.extract_concepts(chunks, n=8) if chunks else []
                st.session_state.tool_output = ("concepts", result)

        if st.button("🗺️ Mind Map", use_container_width=True, disabled=disabled):
            with st.spinner("Building mind map..."):
                chunks = fetch_context(doc_ids)
                result = llm_service.extract_concepts(chunks, n=10) if chunks else []
                st.session_state.tool_output = ("mindmap", result)

        topic_input = st.text_input("Learning guide topic", placeholder="e.g. photosynthesis",
                                     disabled=disabled, key="lg_topic")
        if st.button("📖 Learning Guide", use_container_width=True,
                     disabled=disabled or not topic_input):
            with st.spinner("Creating learning guide..."):
                chunks = fetch_context(doc_ids)
                result = llm_service.learning_guide(topic_input, chunks) if chunks else "No content."
                st.session_state.tool_output = ("guide", result)

        n_quiz = st.slider("Quiz questions", 3, 10, 5, disabled=disabled)
        if st.button("❓ Generate Quiz", use_container_width=True, disabled=disabled):
            with st.spinner("Building quiz..."):
                chunks = fetch_context(doc_ids)
                quiz_data = llm_service.make_quiz(chunks, n=n_quiz) if chunks else []
                st.session_state.tool_output = ("quiz", quiz_data)
                st.session_state.quiz_answers = {}
                st.session_state.progress["quizzes_taken"] += 1

        n_cards = st.slider("Flashcards", 4, 20, 8, disabled=disabled)
        if st.button("🃏 Flashcards", use_container_width=True, disabled=disabled):
            with st.spinner("Creating flashcards..."):
                chunks = fetch_context(doc_ids)
                card_data = llm_service.make_flashcards(chunks, n=n_cards) if chunks else []
                st.session_state.tool_output = ("flashcards", card_data)
                st.session_state.card_idx = 0
                st.session_state.card_flipped = False

        if st.session_state.chat and st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

    # ----- RIGHT: Tool output + Chat -----
    with right:
        if st.session_state.tool_output:
            kind, content = st.session_state.tool_output

            # --- SUMMARY ---
            if kind == "summary":
                with st.expander("📋 Summary", expanded=True):
                    st.markdown(content)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("⬇ Download", content, file_name="summary.txt")
                    with c2:
                        if st.button("📌 Save as note", key="save_summary"):
                            add_note(content, "Summary")
                            st.success("Saved to Notes tab!")

            # --- KEY CONCEPTS ---
            elif kind == "concepts":
                st.markdown("### 🧠 Key Concepts")
                if isinstance(content, dict) and "error" in content:
                    st.error(content.get("raw", "Failed to extract concepts."))
                elif isinstance(content, list) and content:
                    for c in content:
                        with st.expander(f"**{c.get('term', '')}**"):
                            st.markdown(c.get("definition", ""))
                            if c.get("related"):
                                st.caption("Related: " + " · ".join(c["related"]))
                    if st.button("📌 Save concepts as note", key="save_concepts"):
                        note_text = "\n\n".join(
                            [f"**{c.get('term')}**: {c.get('definition')}" for c in content]
                        )
                        add_note(note_text, "Key Concepts")
                        st.success("Saved to Notes tab!")
                else:
                    st.warning("No concepts extracted.")

            # --- MIND MAP ---
            elif kind == "mindmap":
                st.markdown("### 🗺️ Mind Map")
                if isinstance(content, dict) and "error" in content:
                    st.error("Failed to generate mind map.")
                elif isinstance(content, list) and content:
                    render_mind_map(content)
                    st.caption("Hover over nodes to see definitions. Drag to rearrange.")
                else:
                    st.warning("No concepts to map.")

            # --- LEARNING GUIDE ---
            elif kind == "guide":
                with st.expander("📖 Learning Guide", expanded=True):
                    st.markdown(content)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("⬇ Download", content, file_name="learning_guide.txt")
                    with c2:
                        if st.button("📌 Save as note", key="save_guide"):
                            add_note(content, f"Learning Guide: {topic_input}")
                            st.success("Saved to Notes tab!")

            # --- QUIZ ---
            elif kind == "quiz":
                st.markdown("### ❓ Quiz")
                if isinstance(content, dict) and "error" in content:
                    st.error(f"Failed to generate quiz.\n\n{content.get('raw', '')}")
                elif isinstance(content, list) and content:
                    for i, q in enumerate(content):
                        st.markdown(f"**Q{i+1}. {q.get('q', '')}**")
                        options = q.get("options", [])
                        correct = q.get("correct", 0)
                        picked = st.radio("Choose:", options, key=f"quiz_{i}",
                                          index=None, label_visibility="collapsed")
                        if picked is not None:
                            picked_idx = options.index(picked)
                            prev = st.session_state.quiz_answers.get(i)
                            st.session_state.quiz_answers[i] = picked_idx
                            if prev is None:
                                st.session_state.progress["total_mcq_answered"] += 1
                                if picked_idx == correct:
                                    st.session_state.progress["total_mcq_correct"] += 1
                                else:
                                    topic = q.get("topic", "general")
                                    st.session_state.progress["weak_topics"][topic] = \
                                        st.session_state.progress["weak_topics"].get(topic, 0) + 1
                                    st.session_state.progress["missed_questions"].append({
                                        "q": q.get("q", ""),
                                        "correct_answer": options[correct] if correct < len(options) else "",
                                        "topic": topic,
                                    })
                            if picked_idx == correct:
                                st.success(f"✅ Correct! {q.get('explain', '')}")
                            else:
                                correct_text = options[correct] if correct < len(options) else "?"
                                st.error(f"❌ Wrong answer: **{correct_text}** — {q.get('explain', '')}")
                        st.divider()

                    answered = len(st.session_state.quiz_answers)
                    correct_count = sum(
                        1 for i, pick in st.session_state.quiz_answers.items()
                        if pick == content[i].get("correct", -1)
                    )
                    if answered:
                        st.info(f"**Score: {correct_count} / {answered}** answered {answered} of {len(content)}")
                else:
                    st.warning("No questions generated.")

            # --- FLASHCARDS ---
            elif kind == "flashcards":
                st.markdown("### 🃏 Flashcards")
                if isinstance(content, dict) and "error" in content:
                    st.error(f"Failed to generate flashcards.\n\n{content.get('raw', '')}")
                elif isinstance(content, list) and content:
                    idx = st.session_state.get("card_idx", 0)
                    flipped = st.session_state.get("card_flipped", False)
                    card = content[idx]
                    face = card.get("a", "") if flipped else card.get("q", "")
                    side = "Answer" if flipped else "Question"
                    st.markdown(
                        f"""<div style='background:linear-gradient(135deg,#4a6cf7,#6b4af7);color:white;
                        padding:40px 24px;border-radius:12px;min-height:180px;
                        display:flex;flex-direction:column;justify-content:center;
                        text-align:center;margin:16px 0;'>
                        <div style='font-size:0.75rem;opacity:0.7;margin-bottom:12px;
                        text-transform:uppercase;letter-spacing:1px;'>{side}</div>
                        <div style='font-size:1.15rem;line-height:1.5;'>{face}</div>
                        </div>""", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("← Prev", use_container_width=True, disabled=idx == 0):
                            st.session_state.card_idx = idx - 1
                            st.session_state.card_flipped = False
                            st.rerun()
                    with c2:
                        if st.button("🔄 Flip", use_container_width=True, type="primary"):
                            st.session_state.card_flipped = not flipped
                            st.rerun()
                    with c3:
                        if st.button("Next →", use_container_width=True, disabled=idx >= len(content) - 1):
                            st.session_state.card_idx = idx + 1
                            st.session_state.card_flipped = False
                            st.rerun()
                    st.caption(f"Card {idx + 1} of {len(content)}")
                else:
                    st.warning("No flashcards generated.")

        # ----- Chat -----
        st.subheader("Ask your materials")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"<div class='source-box'><b>{s['doc_name']}</b> · p.{s['page']} · score {s['score']}<br>"
                            f"<span style='color:#555'>{s['snippet']}</span></div>",
                            unsafe_allow_html=True)
                if msg["role"] == "assistant" and msg.get("content"):
                    if st.button("📌 Save answer as note", key=f"save_msg_{msg.get('id','x')}"):
                        add_note(msg["content"], "Q&A Answer")
                        st.success("Saved!")

        prompt = st.chat_input(
            "Ask a question about your documents…" if st.session_state.docs else "Upload a document first",
            disabled=not st.session_state.docs)
        if prompt:
            msg_id = str(uuid.uuid4())[:6]
            st.session_state.chat.append({"role": "user", "content": prompt, "id": msg_id})
            st.session_state.progress["questions_asked"] += 1
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = do_ask(prompt, doc_ids)
                st.markdown(res["answer"])
                for s in res.get("sources", []):
                    st.markdown(
                        f"<div class='source-box'><b>{s['doc_name']}</b> · p.{s['page']} · score {s['score']}<br>"
                        f"<span style='color:#555'>{s['snippet']}</span></div>",
                        unsafe_allow_html=True)
                ans_id = str(uuid.uuid4())[:6]
                st.session_state.chat.append({
                    "role": "assistant", "content": res["answer"],
                    "sources": res.get("sources", []), "id": ans_id
                })


# ============================================================
# NOTES TAB
# ============================================================
with notes_tab:
    st.subheader("📝 Saved Notes")
    if not st.session_state.notes:
        st.info("No notes yet. Save summaries, learning guides, key concepts, or chat answers using the 📌 buttons.")
    else:
        all_text = "\n\n---\n\n".join(
            [f"## {n['label']}\n{n['content']}" for n in st.session_state.notes]
        )
        st.download_button("⬇ Download all notes", all_text,
                           file_name="study_genie_notes.txt", mime="text/plain")
        st.markdown(f"**{len(st.session_state.notes)} note(s) saved**")
        st.divider()

        for n in reversed(st.session_state.notes):
            with st.expander(f"📌 {n['label']} (#{n['id']})"):
                st.markdown(n["content"])
                if st.button("🗑 Delete", key=f"del_note_{n['id']}"):
                    st.session_state.notes = [x for x in st.session_state.notes if x["id"] != n["id"]]
                    st.rerun()


# ============================================================
# PROGRESS TAB
# ============================================================
with progress_tab:
    p = st.session_state.progress
    st.subheader("📊 Your Study Progress")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(st.session_state.docs))
    c2.metric("Questions asked", p["questions_asked"])
    c3.metric("Quizzes taken", p["quizzes_taken"])
    accuracy = (p["total_mcq_correct"] / p["total_mcq_answered"] * 100) if p["total_mcq_answered"] else 0
    c4.metric("Quiz accuracy", f"{accuracy:.0f}%" if p["total_mcq_answered"] else "—")

    st.divider()
    st.markdown("### 🎯 Weak Points")
    if p["weak_topics"]:
        sorted_weak = sorted(p["weak_topics"].items(), key=lambda x: x[1], reverse=True)
        st.caption("Topics where you answered incorrectly most often.")
        for topic, count in sorted_weak:
            pct = int((count / max(p["total_mcq_answered"], 1)) * 100)
            st.markdown(
                f"<div style='background:#fef2f2;border-left:3px solid #ef4444;"
                f"padding:10px 14px;margin:6px 0;border-radius:4px;'>"
                f"<b>{topic}</b> — {count} incorrect ({pct}% of answers)</div>",
                unsafe_allow_html=True)
    else:
        st.info("No weak points yet. Take a quiz to start tracking.")

    st.divider()
    st.markdown("### ❌ Questions You Missed")
    missed = p["missed_questions"]
    if missed:
        for i, m in enumerate(reversed(missed[-10:]), 1):
            with st.expander(f"{i}. {m['q'][:80]}{'…' if len(m['q']) > 80 else ''}"):
                st.markdown(f"**Correct answer:** {m['correct_answer']}")
                st.caption(f"Topic: {m['topic']}")
    else:
        st.caption("Missed questions appear here after you take quizzes.")

    st.divider()
    if st.button("🗑 Reset progress"):
        st.session_state.progress = {
            "questions_asked": 0, "quizzes_taken": 0,
            "total_mcq_answered": 0, "total_mcq_correct": 0,
            "weak_topics": {}, "missed_questions": [],
        }
        st.rerun()
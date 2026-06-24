"""Generate answers using Groq (primary) with HuggingFace Inference Providers as fallback."""
import json
import re
from huggingface_hub import InferenceClient
from backend.config import config

# Try Groq first (fast, clean JSON), fall back to HF providers
HF_CHAIN = [
    ("together",  "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    ("together",  "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    ("together",  "deepseek-ai/DeepSeek-V3"),
    ("together",  "deepseek-ai/DeepSeek-R1"),
]

_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None and config.GROQ_API_KEY:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=config.GROQ_API_KEY.strip())
        except Exception as e:
            print(f"[LLM] Groq init failed: {e}", flush=True)
            _groq_client = None
    return _groq_client


RAG_PROMPT = """You are a helpful study assistant. Answer ONLY using the provided context.
If the answer is not in the context, say 'I don't know based on the provided material.'

Context:
{retrieved_chunks}

Question:
{user_query}

Answer:"""

SUMMARY_PROMPT = """Produce a clean, structured summary of the following study material.
Use short sections with bold headings and bullet points. Keep it grounded — do not invent facts.

Material:
{retrieved_chunks}

Summary:"""

QUIZ_ONE_PROMPT = """Based ONLY on the material below, create ONE multiple-choice question.
Return ONLY valid JSON, nothing else:
{{"q": "Question text", "options": ["A", "B", "C", "D"], "correct": 0, "explain": "one-line reason", "topic": "short 2-4 word topic label"}}

The "correct" field is the 0-indexed position of the right answer.
The "topic" field labels the subject area (used for tracking student weak points).
Focus on this aspect: {topic_hint}

Material:
{retrieved_chunks}

JSON:"""

FLASHCARDS_PROMPT_JSON = """Create {n} flashcards from the material below.
Return ONLY valid JSON in this exact format, nothing else:
[
  {{"q": "Question or term", "a": "Short answer or definition"}}
]
Do not invent facts outside the material.

Material:
{retrieved_chunks}

JSON:"""


SYSTEM_MSG = "You output only what the user asks for, in the exact format requested. When asked for JSON, you output valid JSON and nothing else — no explanations, no reasoning, no markdown fences."


def _extract_json(text: str):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        candidate = m.group(0)
    else:
        start = text.find("[")
        if start == -1:
            return None
        candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    last_close = candidate.rfind("}")
    if last_close != -1:
        salvaged = candidate[: last_close + 1] + "]"
        try:
            return json.loads(salvaged)
        except json.JSONDecodeError:
            return None
    return None


def _extract_json_object(text: str):
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _try_generate(prompt: str, max_tokens: int) -> str:
    # 1) Try Groq first
    groq = _get_groq()
    if groq is not None:
        try:
            out = groq.chat.completions.create(
                model=config.GROQ_MODEL.strip(),
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            print(f"[LLM] success via groq / {config.GROQ_MODEL}", flush=True)
            return out.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] groq failed: {e}", flush=True)

    # 2) Fall back to HF Inference Providers
    token = config.HF_TOKEN.strip()
    last_err = None
    for provider, model in HF_CHAIN:
        try:
            client = InferenceClient(api_key=token, provider=provider)
            out = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            print(f"[LLM] success via {provider} / {model}", flush=True)
            return out.choices[0].message.content.strip()
        except Exception as e:
            last_err = f"{provider}/{model}: {e}"
            print(f"[LLM] failed {last_err}", flush=True)
            continue
    return f"[All providers failed. Last error: {last_err}]"


def _format_chunks(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        md = c["metadata"]
        parts.append(
            f"[{i}] (Source: {md.get('doc_name', 'doc')}, p.{md.get('page', '?')})\n{md.get('text', '')}"
        )
    return "\n\n".join(parts)


def answer_question(query: str, chunks) -> str:
    prompt = RAG_PROMPT.format(retrieved_chunks=_format_chunks(chunks), user_query=query)
    return _try_generate(prompt, max_tokens=512)


def summarize(chunks) -> str:
    prompt = SUMMARY_PROMPT.format(retrieved_chunks=_format_chunks(chunks))
    return _try_generate(prompt, max_tokens=700)


def make_quiz(chunks, n: int = 5):
    topics = [
        "main concepts", "definitions", "key relationships",
        "cause and effect", "specific details", "comparisons",
        "processes or steps", "examples given", "limitations or exceptions",
        "practical applications",
    ]
    questions = []
    for i in range(n):
        hint = topics[i % len(topics)]
        prompt = QUIZ_ONE_PROMPT.format(
            retrieved_chunks=_format_chunks(chunks),
            topic_hint=hint,
        )
        raw = _try_generate(prompt, max_tokens=800)
        q = _extract_json_object(raw)
        if q and "q" in q and "options" in q and "correct" in q:
            q.setdefault("topic", hint)
            questions.append(q)
    if not questions:
        return {"error": "Could not generate any questions", "raw": "see logs"}
    return questions


def make_flashcards(chunks, n: int = 8):
    prompt = FLASHCARDS_PROMPT_JSON.format(retrieved_chunks=_format_chunks(chunks), n=n)
    raw = _try_generate(prompt, max_tokens=2500)
    data = _extract_json(raw)
    return data if data else {"error": "Could not parse flashcards", "raw": raw}


CONCEPTS_PROMPT = """Extract the {n} most important key concepts, terms, or ideas from the material below.
Return ONLY valid JSON, nothing else:
[
  {{"term": "concept name", "definition": "one sentence definition", "related": ["related term 1", "related term 2"]}}
]

Material:
{retrieved_chunks}

JSON:"""

LEARNING_GUIDE_PROMPT = """You are a patient, expert tutor. The student wants to deeply understand the topic below.
Create a structured learning guide with:
1. Simple explanation (2-3 sentences, plain language)
2. Core principles (3-5 bullet points)
3. Common misconceptions to avoid
4. 3 follow-up questions the student should be able to answer
Ground everything strictly in the provided material.

Topic: {topic}

Material:
{retrieved_chunks}

Learning Guide:"""


def extract_concepts(chunks, n: int = 8):
    prompt = CONCEPTS_PROMPT.format(retrieved_chunks=_format_chunks(chunks), n=n)
    raw = _try_generate(prompt, max_tokens=1500)
    data = _extract_json(raw)
    return data if data else {"error": "Could not extract concepts", "raw": raw}


def learning_guide(topic: str, chunks) -> str:
    prompt = LEARNING_GUIDE_PROMPT.format(
        topic=topic, retrieved_chunks=_format_chunks(chunks)
    )
    return _try_generate(prompt, max_tokens=1000)
"""
Health Coach Agent — Single file deployment for Render.
No package imports. Everything in one file.
"""
from __future__ import annotations

# ── Patch sqlite3 FIRST before any other import ──────────────────────────────
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import json
import logging
import operator
import os
import random
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from typing_extensions import TypedDict

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
WELLNESS_PDF_PATH = os.environ.get("WELLNESS_PDF_PATH", "./data/wellness_protocol.pdf")
COLLECTION_NAME = "wellness_protocol"


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class HistoryTurn(TypedDict):
    role: str
    content: str
    day: int
    mode: str


class AgentState(TypedDict):
    session_id: str
    mode: str
    patient_profile: Dict[str, Any]
    day_number: int
    session_history: List[HistoryTurn]
    current_input: str
    rag_context: str
    response: str
    error: str


def _empty_state(session_id: str) -> AgentState:
    return {
        "session_id": session_id, "mode": "onboard",
        "patient_profile": {}, "day_number": 1,
        "session_history": [], "current_input": "",
        "rag_context": "", "response": "", "error": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    def __init__(self):
        self._store: Dict[str, AgentState] = {}
        self._lock = threading.Lock()

    def create(self, session_id: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid not in self._store:
                self._store[sid] = _empty_state(sid)
        return sid

    def get(self, session_id: str) -> Optional[AgentState]:
        with self._lock:
            return self._store.get(session_id)

    def update(self, session_id: str, new_state: dict):
        with self._lock:
            if session_id in self._store:
                self._store[session_id].update(new_state)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def __len__(self):
        with self._lock:
            return len(self._store)


session_store = SessionStore()


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class OnboardRequest(BaseModel):
    session_id: Optional[str] = None
    profile_text: str = Field(..., min_length=10)


class CheckInRequest(BaseModel):
    session_id: str
    user_response: Optional[str] = None
    day_number: int = Field(1, ge=1)


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3)


class AgentResponse(BaseModel):
    session_id: str
    response: str
    mode: str
    day_number: int
    patient_name: Optional[str] = None
    error: Optional[str] = None


class SessionStateResponse(BaseModel):
    session_id: str
    patient_profile: Dict[str, Any]
    day_number: int
    history_length: int
    history: List[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# RAG — ChromaDB + Gemini Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _get_chroma_client():
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk_text(text: str, size: int = 600, overlap: int = 100) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _embed(texts: List[str], task: str = "retrieval_document") -> List[List[float]]:
    genai.configure(api_key=GEMINI_API_KEY)
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type=task,
    )
    return result["embedding"] if isinstance(result["embedding"][0], float) else result["embedding"]


def ingest_documents() -> int:
    if not Path(WELLNESS_PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found: {WELLNESS_PDF_PATH}")

    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        logger.info("ChromaDB already has %d chunks.", collection.count())
        return collection.count()

    text = _load_pdf_text(WELLNESS_PDF_PATH)
    chunks = _chunk_text(text)
    embeddings = _embed(chunks, task="retrieval_document")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    logger.info("Ingested %d chunks into ChromaDB.", len(chunks))
    return len(chunks)


def retrieve_context(query: str, n_results: int = 4) -> List[str]:
    if not query.strip():
        return []
    try:
        client = _get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return []
        q_emb = _embed([query], task="retrieval_query")
        results = collection.query(
            query_embeddings=q_emb,
            n_results=min(n_results, collection.count()),
        )
        return [doc for doc in results.get("documents", [[]])[0] if doc]
    except Exception as e:
        logger.error("retrieve_context error: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI LLM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _llm(prompt: str) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a patient onboarding assistant. Extract structured information from the patient's text.

Patient text:
\"\"\"{text}\"\"\"

Return ONLY a valid JSON object with these exact keys (use null for missing fields):
{{
  "name": "<string, default 'Friend' if not mentioned>",
  "age": <integer or null>,
  "primary_goals": ["<goal1>"],
  "sleep_hours": <float or null>,
  "sleep_quality": "<poor|fair|good|excellent or null>",
  "activity_level": "<sedentary|lightly_active|moderately_active|very_active or null>",
  "dietary_restrictions": ["<item>"],
  "health_conditions": ["<condition>"],
  "motivation": "<one sentence>"
}}

No markdown, no explanation. Raw JSON only.
"""

_WELCOME_PROMPT = """\
You are a warm, encouraging health coach. A patient just shared their profile:
{profile_json}

Write a 2-3 sentence warm welcome that:
1. Addresses them by name
2. Reflects back their primary goal(s)
3. Expresses genuine enthusiasm for supporting them

Tone: warm, clear, not clinical, not fluffy. No bullet points.
"""

_DAY_TEMPLATES = {
    "day_1": [
        "How are you feeling right now, physically and emotionally?",
        "What's your biggest hope for this wellness journey?",
        "On a scale of 1-10, how would you rate your energy today?",
    ],
    "day_2_3": [
        "How did Day 1 go? What was easier or harder than expected?",
        "Were you able to follow any of yesterday's goals? Tell me about it.",
        "What's one small win from yesterday you can feel good about?",
    ],
    "day_4_5": [
        "You're nearly halfway through the first week! How are your energy and mood evolving?",
        "Which habit has started to feel a little more automatic?",
        "Is there anything from the protocol you'd like to revisit?",
    ],
    "day_6_7": [
        "You've made it almost a full week — how does that feel?",
        "What surprised you most about yourself this week?",
        "Which habit made the biggest difference, and which needs more work?",
    ],
    "day_8_plus": [
        "How are things going as you continue your journey?",
        "What feels sustainable long-term, and what still feels like a stretch?",
        "What's one thing you'd like to focus on in the days ahead?",
    ],
}

_CHECKIN_PROMPT = """\
You are a warm, empathetic health coach doing a daily check-in.

Patient profile:
- Name: {name}
- Primary goals: {goals}
- Activity level: {activity}
- Sleep quality: {sleep}
- Motivation: {motivation}

Today is Day {day} of their protocol.
Base question: "{question}"
Recent history: {history}

Rewrite the base question to feel personal to THIS patient. 1-2 sentences. Warm, conversational. One focused question only.
"""

_RAG_PROMPT = """\
You are a wellness protocol specialist and health coach.
Answer the patient's question ONLY using the protocol excerpts below.

Rules:
1. If the answer is in the context, give a warm, clear, helpful response (2-4 sentences).
2. If NOT in the context, say: "I don't have specific guidance on that in your protocol. Please consult your healthcare provider."
3. Never invent facts not in the context.
4. Tone: warm, clear, not clinical.

Protocol excerpts:
{context}

Patient name: {name}
Question: {question}
"""


def profile_parser_node(state: AgentState) -> dict:
    try:
        raw = _llm(_EXTRACTION_PROMPT.format(text=state["current_input"]))
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        profile = json.loads(raw)
        welcome = _llm(_WELCOME_PROMPT.format(profile_json=json.dumps(profile, indent=2)))
        return {"patient_profile": profile, "response": welcome, "error": ""}
    except Exception as e:
        logger.exception("profile_parser_node failed")
        return {
            "patient_profile": {"name": "Friend", "primary_goals": [], "motivation": ""},
            "response": "Thanks for sharing! I'm excited to support your wellness journey.",
            "error": str(e),
        }


def day_router_node(state: AgentState) -> dict:
    try:
        day = state.get("day_number", 1)
        profile = state.get("patient_profile", {})
        history = state.get("session_history", [])

        if day == 1: bucket = "day_1"
        elif day <= 3: bucket = "day_2_3"
        elif day <= 5: bucket = "day_4_5"
        elif day <= 7: bucket = "day_6_7"
        else: bucket = "day_8_plus"

        base_q = random.choice(_DAY_TEMPLATES[bucket])
        recent = "\n".join(
            f"  [{t['mode']} Day {t['day']}] {t['role']}: {t['content']}"
            for t in history[-3:]
        ) or "  (no prior check-ins)"

        response = _llm(_CHECKIN_PROMPT.format(
            name=profile.get("name", "Friend"),
            goals=", ".join(profile.get("primary_goals", [])) or "general wellness",
            activity=profile.get("activity_level", "unknown"),
            sleep=profile.get("sleep_quality", "unknown"),
            motivation=profile.get("motivation", "improving health"),
            day=day, question=base_q, history=recent,
        ))
        return {"response": response, "error": ""}
    except Exception as e:
        logger.exception("day_router_node failed")
        return {"response": f"Day {state.get('day_number', 1)} check-in: How are you feeling today?", "error": str(e)}


def rag_answerer_node(state: AgentState) -> dict:
    try:
        query = state["current_input"]
        name = state.get("patient_profile", {}).get("name", "Friend")
        chunks = retrieve_context(query)
        if not chunks:
            return {
                "rag_context": "",
                "response": "I don't have specific guidance on that in your protocol. Please consult your healthcare provider.",
                "error": "",
            }
        context = "\n\n---\n\n".join(chunks)
        response = _llm(_RAG_PROMPT.format(context=context, name=name, question=query))
        return {"rag_context": context, "response": response, "error": ""}
    except Exception as e:
        logger.exception("rag_answerer_node failed")
        return {"rag_context": "", "response": "I'm having trouble accessing the protocol right now. Please try again.", "error": str(e)}


def memory_updater_node(state: AgentState, mode: str, user_input: str, agent_response: str, day: int) -> List[HistoryTurn]:
    turns = []
    if user_input:
        turns.append({"role": "user", "content": user_input, "day": day, "mode": mode})
    if agent_response:
        turns.append({"role": "agent", "content": agent_response, "day": day, "mode": mode})
    return turns


def run_agent(state: AgentState) -> dict:
    """Run the appropriate node based on mode, update memory, return result."""
    mode = state.get("mode", "ask")

    if mode == "onboard":
        result = profile_parser_node(state)
    elif mode == "checkin":
        result = day_router_node(state)
    else:
        result = rag_answerer_node(state)

    new_turns = memory_updater_node(
        state, mode,
        state.get("current_input", ""),
        result.get("response", ""),
        state.get("day_number", 1),
    )

    result["session_history"] = state.get("session_history", []) + new_turns
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        n = ingest_documents()
        logger.info("RAG ready — %d chunks in ChromaDB.", n)
    except FileNotFoundError:
        logger.warning("Wellness PDF not found. RAG will be unavailable.")
    except Exception as e:
        logger.error("RAG ingestion failed: %s", e)
    yield


app = FastAPI(
    title="Health Coach Agent",
    description="AI health coach: onboarding, daily check-ins, protocol Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Health Coach Agent</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f9fafb; }
    h1 { color: #1a7f5a; }
    .card { background: white; border-radius: 12px; padding: 24px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    input, textarea { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; box-sizing: border-box; }
    button { background: #1a7f5a; color: white; border: none; padding: 10px 24px; border-radius: 8px; cursor: pointer; font-size: 15px; }
    button:hover { background: #15694a; }
    .response { background: #f0fdf4; border-left: 4px solid #1a7f5a; padding: 16px; border-radius: 8px; margin-top: 16px; white-space: pre-wrap; }
    .label { font-weight: bold; color: #374151; margin-top: 12px; display: block; }
    .sid { font-size: 12px; color: #6b7280; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>🌿 Health Coach Agent</h1>

  <div class="card">
    <h2>Step 1: Onboard</h2>
    <textarea id="profile" rows="4" placeholder="Tell me about yourself: name, age, goals, sleep habits, activity level..."></textarea>
    <button onclick="onboard()">Start My Wellness Journey</button>
    <div class="sid" id="sid"></div>
    <div class="response" id="onboard-resp" style="display:none"></div>
  </div>

  <div class="card">
    <h2>Step 2: Daily Check-In</h2>
    <span class="label">Day number:</span>
    <input type="number" id="day" value="1" min="1" style="width:80px">
    <textarea id="user-response" rows="3" placeholder="Your response to yesterday's check-in (optional)..."></textarea>
    <button onclick="checkin()">Do Check-In</button>
    <div class="response" id="checkin-resp" style="display:none"></div>
  </div>

  <div class="card">
    <h2>Step 3: Ask Protocol Questions</h2>
    <input type="text" id="question" placeholder="e.g. How many hours of sleep should I get?">
    <button onclick="ask()">Ask</button>
    <div class="response" id="ask-resp" style="display:none"></div>
  </div>

  <script>
    let sessionId = null;

    async function onboard() {
      const text = document.getElementById('profile').value.trim();
      if (!text) return alert('Please enter your profile text.');
      const res = await fetch('/api/onboard', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({profile_text: text})
      });
      const data = await res.json();
      sessionId = data.session_id;
      document.getElementById('sid').textContent = 'Session ID: ' + sessionId;
      show('onboard-resp', data.response || data.detail);
    }

    async function checkin() {
      if (!sessionId) return alert('Please onboard first.');
      const day = parseInt(document.getElementById('day').value);
      const resp = document.getElementById('user-response').value.trim();
      const res = await fetch('/api/checkin', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, user_response: resp || null, day_number: day})
      });
      const data = await res.json();
      show('checkin-resp', data.response || data.detail);
    }

    async function ask() {
      if (!sessionId) return alert('Please onboard first.');
      const q = document.getElementById('question').value.trim();
      if (!q) return alert('Please enter a question.');
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, question: q})
      });
      const data = await res.json();
      show('ask-resp', data.response || data.detail);
    }

    function show(id, text) {
      const el = document.getElementById(id);
      el.style.display = 'block';
      el.textContent = text;
    }
  </script>
</body>
</html>
""")


@app.post("/api/onboard", response_model=AgentResponse)
async def onboard(req: OnboardRequest):
    sid = session_store.create(req.session_id)
    state = session_store.get(sid)
    state["mode"] = "onboard"
    state["current_input"] = req.profile_text
    result = run_agent(state)
    session_store.update(sid, result)
    return AgentResponse(
        session_id=sid,
        response=result["response"],
        mode="onboard",
        day_number=state.get("day_number", 1),
        patient_name=result.get("patient_profile", {}).get("name"),
        error=result.get("error") or None,
    )


@app.post("/api/checkin", response_model=AgentResponse)
async def checkin(req: CheckInRequest):
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please onboard first.")
    state = session_store.get(req.session_id)
    state["mode"] = "checkin"
    state["day_number"] = req.day_number
    state["current_input"] = req.user_response or ""
    result = run_agent(state)
    session_store.update(req.session_id, result)
    return AgentResponse(
        session_id=req.session_id,
        response=result["response"],
        mode="checkin",
        day_number=req.day_number,
        patient_name=state.get("patient_profile", {}).get("name"),
        error=result.get("error") or None,
    )


@app.post("/api/ask", response_model=AgentResponse)
async def ask(req: AskRequest):
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please onboard first.")
    state = session_store.get(req.session_id)
    state["mode"] = "ask"
    state["current_input"] = req.question
    result = run_agent(state)
    session_store.update(req.session_id, result)
    return AgentResponse(
        session_id=req.session_id,
        response=result["response"],
        mode="ask",
        day_number=state.get("day_number", 1),
        patient_name=state.get("patient_profile", {}).get("name"),
        error=result.get("error") or None,
    )


@app.get("/api/state/{session_id}", response_model=SessionStateResponse)
async def get_state(session_id: str):
    state = session_store.get(session_id)
    if state is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found."})
    return SessionStateResponse(
        session_id=session_id,
        patient_profile=state.get("patient_profile", {}),
        day_number=state.get("day_number", 1),
        history_length=len(state.get("session_history", [])),
        history=state.get("session_history", []),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(session_store)}

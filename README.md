# Health Coach Agent

An AI health coach built with FastAPI + Google Gemini + ChromaDB (RAG).

## Features
- Patient onboarding with profile parsing
- Daily adaptive check-ins (Day 1 → Day 8+)
- Protocol Q&A grounded in a wellness PDF (RAG)
- Session memory across check-ins

## Deployment on Render

1. Push this repo to GitHub
2. Create a new Web Service on Render pointing to this repo
3. Set environment variable: `GEMINI_API_KEY`
4. Build Command: `pip install -r requirements.txt && python generate_pdf.py`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/onboard` | Onboard a new patient |
| POST | `/api/checkin` | Daily check-in |
| POST | `/api/ask` | Ask a protocol question |
| GET | `/api/state/{session_id}` | Get session state |
| GET | `/health` | Health check |

## Tech Stack
- **FastAPI** — Web framework
- **Google Gemini** — LLM (gemini-1.5-flash) + Embeddings
- **ChromaDB** — Vector store for RAG
- **FPDF2** — PDF generation

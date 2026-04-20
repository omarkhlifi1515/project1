# ENISO Enterprise Assistant Runbook

## Services
- API: `uvicorn backend.api.main:app --reload`
- Frontend: `cd frontend && npm run dev`
- Optional full stack: `docker compose up --build`

## First-Time Setup
1. Install Python dependencies: `pip install -r requirements.txt`
2. Install frontend dependencies: `cd frontend && npm install`
3. Pull local model: `ollama pull qwen2.5:3b`
4. Start Ollama server: `ollama serve`

## Authentication Bootstrap
- Register admin user using `/api/auth/register` and set `"role":"admin"` in request body.
- Login with `/api/auth/login` to get bearer token.

## Indexing Operations
- Manual sync endpoint: `POST /api/admin/sync-index`
- Full refresh job endpoint: `POST /api/admin/jobs/refresh`
- Job status endpoint: `GET /api/admin/jobs/status`

## Monitoring
- Health: `GET /api/health`
- Metrics: `GET /metrics`

## Incident Checklist
1. Verify Ollama is running and model exists.
2. Check API logs for auth/token errors.
3. Run refresh job and verify `processed_data/` regenerated.
4. Confirm Chroma folders are writable.

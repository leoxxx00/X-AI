# Unified QA Platform

Production-ready URL Q/A evaluator and generator.

## Services
- API: FastAPI
- Worker: Python background processor
- Web: Next.js
- Storage: Postgres + Redis

## Local run
1. Copy .env.example to .env
2. Fill DEEPSEEK_API_KEY
3. Start:
   docker compose -f infra/docker-compose.yml up --build

## URLs
- Web: http://localhost:3000
- API: http://localhost:8000
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
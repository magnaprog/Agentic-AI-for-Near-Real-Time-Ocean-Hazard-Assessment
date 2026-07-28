# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python backend serving static files
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY backend/ backend/

# Copy built frontend into the location the backend expects
COPY --from=frontend-build /app/frontend/dist frontend/dist

RUN chown -R app:app /app
USER app

EXPOSE 8100
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8100"]

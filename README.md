# wifi-fall Backend

FastAPI backend for an intelligent Wi-Fi CSI fall detection simulation system.

## Requirements

- Python 3.10+

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
```

Default health check:

```text
GET http://127.0.0.1:8000/api/health
```

## Test

```bash
pytest
```

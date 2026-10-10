<div align="center">

# AI Receptionist

**A voice agent that answers the phone for a small business, books appointments, and hands the call to a human when it should.**

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-Supabase-3ECF8E?logo=supabase&logoColor=white)
![React](https://img.shields.io/badge/dashboard-React%20%2B%20Vite-61DAFB?logo=react&logoColor=black)
![Bring your own keys](https://img.shields.io/badge/API%20keys-bring%20your%20own-orange)

</div>

---

## What it does

A caller talks to the receptionist the way they would to a person. It checks real availability, books, moves or cancels
an appointment, and transfers to a colleague when asked. Everything it says about time comes from the database, never
from the model's imagination.

- **Live voice**: streaming speech-to-text, a tool-using LLM, and streaming text-to-speech, with barge-in (talk over it
  and it stops).
- **Safe booking**: the database itself refuses a double booking, even under concurrent requests.
- **Natural dates**: "next Tuesday evening" is resolved in the business's own timezone by code, not by the model.
- **Three ways in**: terminal chat, a browser microphone demo, and a Twilio phone line.
- **Dashboard**: a one-screen view of calls and bookings (`frontend/`).

## How it works

```
 caller ──► Twilio / browser mic ──► speech-to-text ──► agent loop ──► text-to-speech ──► caller
                                                           │
                                                  tools: check_availability,
                                                  book / move / cancel, handoff
                                                           │
                                                      Postgres
```

| Layer | Where | Notes |
| --- | --- | --- |
| Agent loop and tools | `backend/app/agent/` | Hand-written streaming loop; strict tool schemas |
| Booking logic | `backend/app/domain/` | Slots, availability, date parsing |
| Voice pipeline | `backend/app/voice/` | Deepgram or Groq speech, barge-in |
| Phone | `backend/app/telephony/`, `backend/app/api/twilio.py` | Twilio webhook and Media Streams |
| Data | `backend/app/models.py`, `backend/alembic/` | Alembic migrations, GiST exclusion constraint |
| Dashboard | `frontend/` | React, TypeScript, Vite |

Design decisions worth knowing:

1. **The model cannot invent a time.** `book_appointment` takes a `slot_id`, not a time, so a made-up appointment is
   unrepresentable.
2. **The database is the last line of defence.** A Postgres exclusion constraint blocks overlapping bookings.
3. **Every provider is swappable by config.** The LLM endpoint is OpenAI-compatible; speech providers are chosen by
   environment variable.

---

## Bring your own keys

This project ships with **no keys and no shared accounts**. You create your own free accounts and the setup script asks
for the values. Nothing leaves your machine except calls to the providers you chose.

| Service | Needed for | Required | Get it |
| --- | --- | --- | --- |
| **Supabase** (Postgres) | Storing businesses, bookings, calls | Yes | Project, then Connect, then Session pooler string |
| **Groq** | The LLM (and keyless speech fallback) | Yes | console.groq.com/keys |
| **Deepgram** | Faster, better streaming speech | Recommended | console.deepgram.com |
| **Twilio** | A real phone number | Optional | twilio.com/console |

Your keys are written to a local `.env` file, which is git-ignored. Never commit it.

---

## Quick start

**Prerequisites:** Python 3.12+, Node 20+ (dashboard only), and the keys above.

### 1. Run the setup script

Windows (PowerShell):

```powershell
./setup.ps1
```

macOS / Linux:

```bash
./setup.sh
```

The script creates the virtual environment, installs dependencies, asks for your keys, writes `.env`, applies the
database migrations and seeds a demo business.

### 2. Start the server

```bash
python -m backend.app.serve
```

Open <http://127.0.0.1:8000> for the browser voice demo, or check <http://127.0.0.1:8000/health>.

### 3. Or talk to it in the terminal

```bash
python -m backend.app.cli.chat
```

Type `quit` to hang up.

### 4. Dashboard (optional)

```bash
cd frontend
npm install
npm run dev
```

---

## Manual setup

Prefer to do it by hand? The script does exactly this.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m scripts.setup_env deps > requirements.txt && pip install -r requirements.txt
cp .env.example .env          # then fill in your keys (or: python -m scripts.setup_env env)
alembic upgrade head
python -m scripts.seed_demo
```

### Configuration

Set in `.env` (see `.env.example` for every option, with comments):

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Supabase Session pooler string, with the scheme changed to `postgresql+psycopg://` |
| `LLM_API_KEY` | Your Groq key |
| `LLM_BASE_URL`, `LLM_MODEL` | Point at any OpenAI-compatible provider |
| `DEEPGRAM_API_KEY` | Enables Deepgram speech; without it the Groq fallback is used |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` | Real phone calls |
| `PUBLIC_BASE_URL` | Public HTTPS address of your server, needed by Twilio |

If your database password contains `@ : / ? # %`, percent-encode it, otherwise the URL parses as a different host.

### Real phone calls

Twilio must reach your server over the internet, so expose it (for example with a Cloudflare quick tunnel), set
`PUBLIC_BASE_URL` to that address, restart the server, and point your Twilio number's voice webhook at
`/twilio/voice`. The number callers dial must match the business seeded with `--phone`.

```bash
python -m scripts.seed_demo --reset --phone +15551234567 --handoff-phone +919876543210
```

---

## Testing

```bash
pytest -m "not db"     # fast tests, no database needed
pytest                 # full suite, needs DATABASE_URL
```

## Known limitations

- **Twilio trial accounts cannot run the live phone agent.** Trial accounts block the `<Connect><Stream>` and
  `<Dial>` verbs, so audio streaming and human transfer need a paid Twilio account. The protocol is covered by the
  simulator (`scripts/twilio_sim.py`) and tests; the browser demo works without Twilio.
- **Free LLM tiers are rate limited.** Groq's free key allows roughly 8K tokens per minute, so sustained conversations
  can stall on rate-limit retries. A second provider via `LLM_BASE_URL` is the remedy.
- **Database latency.** A distant Supabase region adds hundreds of milliseconds per query and slows the database tests.
- **Windows:** start the server with `python -m backend.app.serve` (not bare `uvicorn`), because psycopg's async mode
  needs a selector event loop.

## Project layout

```
backend/    FastAPI app, agent, voice pipeline, telephony, migrations
frontend/   React dashboard
scripts/    seeding, smoke tests, Twilio and voice helpers
tests/      unit and database tests
```

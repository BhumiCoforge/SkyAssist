# SkyAssist: AI-Powered Airline Support and Flight Booking Assistant

A website where customers search flights, book and manage tickets, and chat with an AI assistant that answers airline policy questions from documents and performs actions after the customer confirms.

## Modules

| # | Module | Where |
|---|---|---|
| 1 | Flight management (search by route, date, price, seats; CRUD) | `main.py`, `services.py` |
| 2 | Booking management (book, view, update passengers, cancel with refund) | `main.py`, `services.py`, `db.py` |
| 3 | AI chatbot (natural language, remembers the conversation) | `agent.py` |
| 4 | RAG airline support (chunking, embeddings, FAISS, cited answers) | `rag.py`, `docs/` |
| 5 | AI agent with confirmation workflow | `agent.py` |
| UI | Home, results, booking, confirmation, chat, my bookings | `frontend/index.html` |

## How a request flows

1. The customer types a message and the browser calls `POST /chat`.
2. The LangChain agent decides which tools to call. One message can use several, for example `search_flights` and `policy_lookup` together.
3. `search_flights` reads the SQLite database through SQLAlchemy. `policy_lookup` retrieves chunks from the FAISS index built from `docs/`.
4. The LLM combines the results into one answer. The frontend shows flight cards and policy sources inside the chat.
5. `request_booking` and `request_cancellation` only create a pending action. The customer presses Confirm in the chat (`POST /chat/confirm/{id}`), and only then does the database change.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
# open .env and set ANTHROPIC_API_KEY
uvicorn main:app --reload
```

- Website: http://127.0.0.1:8000
- API docs (Swagger, for testing): http://127.0.0.1:8000/docs

If you have an older `skyassist.db`, delete it once, because the tables changed. Sample flights for the next 45 days are created automatically on first start. The first policy question downloads the embedding model (internet needed once) and builds the index.

To use OpenAI or Gemini, install `langchain-openai` or `langchain-google-genai` and set `LLM_PROVIDER` and `LLM_MODEL` in `.env`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/flights?origin=&destination=&date=YYYY-MM-DD&max_price=&passengers=` | Search flights |
| POST, GET, PUT, DELETE | `/flights`, `/flights/{id}` | Flight CRUD |
| POST | `/bookings` | Create booking (customer, passengers, total price, seat check) |
| GET | `/bookings/{id}`, `/bookings?email=` | Check bookings |
| PUT | `/bookings/{id}` | Update passenger information |
| GET | `/bookings/{id}/cancel-quote` | Preview refund |
| DELETE | `/bookings/{id}` | Cancel booking and release seats |
| POST | `/chat` | Message the AI assistant |
| POST | `/chat/confirm/{id}`, `/chat/decline/{id}` | Confirm or decline an AI action |
| POST | `/rag/rebuild` | Rebuild the index after editing `docs/` |

Cancellation refunds follow `docs/refunds.md`: more than 72h before departure gives the fare minus INR 500, 24 to 72h gives 50%, under 24h gives nothing.

## Demo script

1. Home page: search Delhi to Mumbai, pick a flight, book it with two passengers, then open My Bookings and edit a name.
2. Chat: "Hi, I want to book a flight from Delhi to Mumbai on 20 December. Show me flights under ₹7,000."
3. Chat: "What is the baggage allowance?" shows a cited policy answer.
4. Chat: "Book the second flight for me." Give a name and email, review the summary and press Confirm.
5. Chat: "Cancel booking 1" shows the refund, then Confirm.
6. Chat: "Find flights from Delhi to Goa tomorrow and tell me the cancellation charges" shows two tools in one answer.

## Testing checklist

- Book more seats than available (expect an error)
- Book with a missing name or invalid email
- Cancel twice (second attempt is rejected)
- Check that seats go down on booking and back up on cancel
- Restart the server and confirm bookings persist
- Ask something not covered by the policy documents (assistant should say it doesn't know)

## Known limitations

- Pending confirmations and chat memory are in memory and reset on restart.
- No login or payment. Bookings are looked up by booking ID.
- Policy documents and flights are sample data. Add real PDFs to `docs/` and call `/rag/rebuild`.

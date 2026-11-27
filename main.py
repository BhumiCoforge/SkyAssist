import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from sqlalchemy import inspect, text

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import services as svc
from config import settings
from db import Base, engine, seed
from routers import bookings, chat, flights
from payments import PaymentError, expire_stale_bookings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("skyassist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    # Non-destructive SQLite compatibility migration: add nullable expiry column if missing.
    inspector = inspect(engine)
    if "bookings" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("bookings")}
        missing = []
        if "payment_expires_at" not in cols:
            missing.append("ALTER TABLE bookings ADD COLUMN payment_expires_at DATETIME")
        if "idempotency_key" not in cols:
            missing.append("ALTER TABLE bookings ADD COLUMN idempotency_key VARCHAR(128)")
        if missing:
            with engine.begin() as conn:
                for statement in missing:
                    conn.execute(text(statement))
        # Idempotency index for upgraded SQLite databases (create_all cannot alter existing tables).
        with engine.begin() as conn:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_bookings_idempotency_key ON bookings (idempotency_key)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_bookings_payment_expires_at ON bookings (payment_expires_at)"))
    expire_stale_bookings()
    added = seed()   # tops up missing days, so flights never run out
    if added:
        log.info("seeded %s flights", added)
    async def payment_expiry_worker():
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(expire_stale_bookings)

    worker = asyncio.create_task(payment_expiry_worker())
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.exception_handler(PaymentError)
async def payment_error_handler(request, exc: PaymentError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.exception_handler(svc.BookingError)
async def booking_error_handler(request, exc: svc.BookingError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


app.include_router(flights.router)
app.include_router(bookings.router)
app.include_router(chat.router)

# Frontend (must be mounted last)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")
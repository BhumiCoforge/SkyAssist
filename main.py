import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import services as svc
from config import settings
from db import Base, engine, seed
from routers import bookings, chat, flights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("skyassist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    added = seed()   # tops up missing days, so flights never run out
    if added:
        log.info("seeded %s flights", added)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


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
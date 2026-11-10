import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

import services as svc
from db import Base, Booking, Customer, Flight, engine, get_db, seed

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    seed()
    yield


app = FastAPI(title=os.getenv("APP_NAME", "SkyAssist API"), lifespan=lifespan)


@app.exception_handler(svc.BookingError)
async def booking_error_handler(request, exc: svc.BookingError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


# ---------- Schemas ----------
class FlightIn(BaseModel):
    flight_no: str = Field(min_length=2)
    airline: str = Field(min_length=2)
    origin: str = Field(min_length=2)
    destination: str = Field(min_length=2)
    departure: datetime
    arrival: datetime
    price: float = Field(gt=0)
    seats: int = Field(default=50, ge=0)


class PassengerIn(BaseModel):
    name: str = Field(min_length=2)
    age: Optional[int] = Field(default=None, ge=0, le=120)


class CustomerIn(BaseModel):
    name: str = Field(min_length=2)
    email: EmailStr
    phone: str = ""


class BookingIn(BaseModel):
    flight_id: int
    customer: CustomerIn
    passengers: list[PassengerIn] = Field(min_length=1, max_length=9)


class PassengersUpdate(BaseModel):
    passengers: list[PassengerIn] = Field(min_length=1, max_length=9)


class ChatIn(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


# ---------- Module 1: Flights ----------
@app.get("/flights")
def search_flights(origin: str = "", destination: str = "", date: str = "", max_price: float = 0,
                   passengers: int = 1, db: Session = Depends(get_db)):
    return svc.search_flights(db, origin, destination, date, max_price, passengers)


@app.post("/flights", status_code=201)
def create_flight(flight: FlightIn, db: Session = Depends(get_db)):
    f = Flight(**flight.model_dump())
    db.add(f)
    db.commit()
    return svc.flight_dict(f)


@app.get("/flights/{flight_id}")
def get_flight(flight_id: int, db: Session = Depends(get_db)):
    f = db.get(Flight, flight_id)
    if not f:
        raise HTTPException(404, "Flight not found")
    return svc.flight_dict(f)


@app.put("/flights/{flight_id}")
def update_flight(flight_id: int, flight: FlightIn, db: Session = Depends(get_db)):
    f = db.get(Flight, flight_id)
    if not f:
        raise HTTPException(404, "Flight not found")
    for k, v in flight.model_dump().items():
        setattr(f, k, v)
    db.commit()
    return svc.flight_dict(f)


@app.delete("/flights/{flight_id}")
def delete_flight(flight_id: int, db: Session = Depends(get_db)):
    f = db.get(Flight, flight_id)
    if not f:
        raise HTTPException(404, "Flight not found")
    if db.query(Booking).filter_by(flight_id=flight_id).first():
        raise HTTPException(409, "Flight has bookings and cannot be deleted")
    db.delete(f)
    db.commit()
    return {"message": "Flight deleted"}


# ---------- Module 2: Bookings ----------
@app.post("/bookings", status_code=201)
def create_booking(body: BookingIn, db: Session = Depends(get_db)):
    c = body.customer
    return svc.create_booking(db, body.flight_id, c.name, c.email, c.phone,
                              [p.model_dump() for p in body.passengers])


@app.get("/bookings")
def list_bookings(email: str = "", db: Session = Depends(get_db)):
    q = db.query(Booking)
    if email:
        q = q.join(Customer).filter(Customer.email == email.strip().lower())
    return [svc.booking_dict(b) for b in q.order_by(Booking.id.desc()).limit(50)]


@app.get("/bookings/{booking_id}")
def get_booking(booking_id: int, db: Session = Depends(get_db)):
    return svc.booking_dict(svc.get_booking(db, booking_id))


@app.put("/bookings/{booking_id}")
def update_booking(booking_id: int, body: PassengersUpdate, db: Session = Depends(get_db)):
    return svc.update_passengers(db, booking_id, [p.model_dump() for p in body.passengers])


@app.get("/bookings/{booking_id}/cancel-quote")
def cancel_quote(booking_id: int, db: Session = Depends(get_db)):
    return svc.refund_quote(svc.get_booking(db, booking_id))


@app.delete("/bookings/{booking_id}")
def cancel_booking(booking_id: int, db: Session = Depends(get_db)):
    return svc.cancel_booking(db, booking_id)


# ---------- Modules 3-5: Chat / RAG / Agent (imported lazily so CRUD works without an API key) ----------
@app.post("/chat")
def chat_endpoint(body: ChatIn):
    from agent import chat

    try:
        return chat(body.session_id, body.message)
    except Exception as e:
        raise HTTPException(500, f"Assistant error: {e}")


@app.post("/chat/confirm/{pid}")
def chat_confirm(pid: str):
    from agent import confirm

    return confirm(pid)


@app.post("/chat/decline/{pid}")
def chat_decline(pid: str):
    from agent import decline

    return decline(pid)


@app.post("/rag/rebuild")
def rebuild_index():
    from rag import build_index

    return {"chunks_indexed": build_index()}


# ---------- Frontend (must be mounted last) ----------
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")

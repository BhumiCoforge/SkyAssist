"""Business logic shared by the REST API and the AI agent."""
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import settings
from db import Booking, BookingStatus, Customer, Flight, Passenger

log = logging.getLogger("skyassist.services")

CANCEL_FEE = settings.cancel_fee

# Customers say "Bangalore", "Bombay" or "BLR"; the database stores one official name per city.
CITY_ALIASES = {
    "bombay": "Mumbai", "bom": "Mumbai",
    "new delhi": "Delhi", "del": "Delhi",
    "bangalore": "Bengaluru", "blr": "Bengaluru",
    "madras": "Chennai", "maa": "Chennai",
    "calcutta": "Kolkata", "ccu": "Kolkata",
    "hyd": "Hyderabad", "goi": "Goa", "lko": "Lucknow",
}


class BookingError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def normalize_city(name: str) -> str:
    name = (name or "").strip()
    return CITY_ALIASES.get(name.lower(), name)


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def flight_dict(f: Flight) -> dict:
    return dict(id=f.id, flight_no=f.flight_no, airline=f.airline, origin=f.origin,
                destination=f.destination, departure=f.departure.isoformat(),
                arrival=f.arrival.isoformat(), price=f.price, seats=f.seats)


def booking_dict(b: Booking) -> dict:
    return dict(id=b.id, status=b.status, num_seats=b.num_seats, total_price=b.total_price,
                created_at=b.created_at.isoformat(), flight=flight_dict(b.flight),
                customer=dict(name=b.customer.name, email=b.customer.email, phone=b.customer.phone),
                passengers=[dict(name=p.name, age=p.age) for p in b.passengers])


def search_flights(db: Session, origin="", destination="", date="", max_price=0, passengers=1,
                   limit: int = 20, offset: int = 0) -> list:
    q = db.query(Flight).filter(Flight.seats >= max(passengers, 1))
    if origin:
        q = q.filter(Flight.origin.ilike(normalize_city(origin)))
    if destination:
        q = q.filter(Flight.destination.ilike(normalize_city(destination)))
    if max_price:
        q = q.filter(Flight.price <= max_price)
    start = datetime.now()
    if date:
        try:
            day = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise BookingError("Date must be in YYYY-MM-DD format")
        q = q.filter(Flight.departure >= max(day, start), Flight.departure < day + timedelta(days=1))
    else:
        q = q.filter(Flight.departure >= start)
    return [flight_dict(f) for f in q.order_by(Flight.departure).offset(offset).limit(limit)]


def _clean_passengers(passengers: list[dict]) -> list[dict]:
    out = [dict(name=p["name"].strip(), age=p.get("age")) for p in passengers if p.get("name", "").strip()]
    if not 1 <= len(out) <= 9:
        raise BookingError("Provide between 1 and 9 passengers with names")
    return out


def check_bookable(db: Session, flight_id: int, n: int) -> Flight:
    f = db.get(Flight, flight_id)
    if not f:
        raise BookingError(f"Flight {flight_id} not found", 404)
    if f.departure <= datetime.now():
        raise BookingError("This flight has already departed")
    if f.seats < n:
        raise BookingError(f"Only {f.seats} seat(s) left on {f.flight_no}")
    return f


def create_booking(db: Session, flight_id: int, name: str, email: str, phone: str,
                   passengers: list[dict]) -> dict:
    passengers = _clean_passengers(passengers)
    f = check_bookable(db, flight_id, len(passengers))
    email = normalize_email(email)
    try:
        cust = db.query(Customer).filter_by(email=email).first()
        if cust is None:
            cust = Customer(name=name.strip(), email=email, phone=phone or "")
            db.add(cust)
            db.flush()
        elif phone and not cust.phone:
            cust.phone = phone   # only fill a blank; never overwrite an existing customer's details

        # Atomic seat reservation: two people booking the last seat at once cannot both succeed.
        res = db.execute(update(Flight)
                         .where(Flight.id == f.id, Flight.seats >= len(passengers))
                         .values(seats=Flight.seats - len(passengers)))
        if res.rowcount != 1:
            db.rollback()
            raise BookingError("Those seats were just taken. Please search again.", 409)

        b = Booking(flight_id=f.id, customer_id=cust.id, num_seats=len(passengers),
                    total_price=round(f.price * len(passengers), 2))
        b.passengers = [Passenger(**p) for p in passengers]
        db.add(b)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise BookingError("Could not save the booking. Please try again.", 409)
    db.refresh(f)
    log.info("booking created id=%s flight=%s seats=%s", b.id, f.flight_no, b.num_seats)
    return booking_dict(b)


def get_booking(db: Session, booking_id: int) -> Booking:
    """Internal lookup with no ownership check. Customer-facing code must use get_owned_booking."""
    b = db.get(Booking, booking_id)
    if not b:
        raise BookingError(f"Booking {booking_id} not found", 404)
    return b


def get_owned_booking(db: Session, booking_id: int, email: str) -> Booking:
    """Return the booking only if `email` is the one it was booked with.
    The same error is used for 'missing' and 'wrong email' so ids cannot be guessed."""
    b = db.get(Booking, booking_id)
    if not b or not email or b.customer.email != normalize_email(email):
        raise BookingError("No booking found with that ID and email", 404)
    return b


def refund_quote(b: Booking) -> dict:
    hours = (b.flight.departure - datetime.now()).total_seconds() / 3600
    if b.status == BookingStatus.PENDING_PAYMENT.value:
        return dict(refund=0, rule="Payment not completed; no refund is due")
    if b.status == BookingStatus.CANCELLED.value:
        return dict(refund=0, rule="Already cancelled")
    if hours <= 0:
        return dict(refund=0, rule="Flight has already departed")
    if hours > 72:
        return dict(refund=max(b.total_price - CANCEL_FEE, 0),
                    rule=f"More than 72h before departure: INR {CANCEL_FEE} fee")
    if hours >= 24:
        return dict(refund=round(b.total_price * 0.5, 2), rule="24-72h before departure: 50% refund")
    return dict(refund=0, rule="Less than 24h before departure: no refund")


def cancel_booking(db: Session, booking_id: int, email: str) -> dict:
    b = get_owned_booking(db, booking_id, email)
    if b.status == BookingStatus.CANCELLED.value:
        raise BookingError("Booking is already cancelled")
    if b.flight.departure <= datetime.now():
        raise BookingError("This flight has already departed and cannot be cancelled")
    quote = refund_quote(b)
    # Only one request can flip CONFIRMED -> CANCELLED, so seats are never released twice.
    res = db.execute(update(Booking)
                     .where(Booking.id == b.id, Booking.status == BookingStatus.CONFIRMED.value)
                     .values(status=BookingStatus.CANCELLED.value))
    if res.rowcount != 1:
        db.rollback()
        raise BookingError("Booking is already cancelled")
    db.execute(update(Flight).where(Flight.id == b.flight_id).values(seats=Flight.seats + b.num_seats))
    db.commit()
    db.refresh(b)
    db.refresh(b.flight)
    log.info("booking cancelled id=%s refund=%s", b.id, quote["refund"])
    return dict(booking=booking_dict(b), **quote)


def update_passengers(db: Session, booking_id: int, email: str, passengers: list[dict]) -> dict:
    b = get_owned_booking(db, booking_id, email)
    if b.status != BookingStatus.CONFIRMED.value:
        raise BookingError("Only confirmed bookings can be updated")
    if b.flight.departure <= datetime.now():
        raise BookingError("This flight has already departed")
    passengers = _clean_passengers(passengers)
    if len(passengers) != b.num_seats:
        raise BookingError(f"This booking has {b.num_seats} passenger(s); the count cannot change")
    b.passengers = [Passenger(**p) for p in passengers]
    db.commit()
    return booking_dict(b)
"""Business logic shared by the REST API and the AI agent."""
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from db import Booking, Customer, Flight, Passenger

CANCEL_FEE = 500


class BookingError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def flight_dict(f: Flight) -> dict:
    return dict(id=f.id, flight_no=f.flight_no, airline=f.airline, origin=f.origin,
                destination=f.destination, departure=f.departure.isoformat(),
                arrival=f.arrival.isoformat(), price=f.price, seats=f.seats)


def booking_dict(b: Booking) -> dict:
    return dict(id=b.id, status=b.status, num_seats=b.num_seats, total_price=b.total_price,
                created_at=b.created_at.isoformat(), flight=flight_dict(b.flight),
                customer=dict(name=b.customer.name, email=b.customer.email, phone=b.customer.phone),
                passengers=[dict(name=p.name, age=p.age) for p in b.passengers])


def search_flights(db: Session, origin="", destination="", date="", max_price=0, passengers=1) -> list:
    q = db.query(Flight).filter(Flight.seats >= max(passengers, 1))
    if origin:
        q = q.filter(Flight.origin.ilike(origin.strip()))
    if destination:
        q = q.filter(Flight.destination.ilike(destination.strip()))
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
    return [flight_dict(f) for f in q.order_by(Flight.departure).limit(20)]


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


def create_booking(db: Session, flight_id: int, name: str, email: str, phone: str, passengers: list[dict]) -> dict:
    passengers = _clean_passengers(passengers)
    f = check_bookable(db, flight_id, len(passengers))
    email = email.strip().lower()
    cust = db.query(Customer).filter_by(email=email).first()
    if cust:
        cust.name, cust.phone = name.strip(), phone or cust.phone
    else:
        cust = Customer(name=name.strip(), email=email, phone=phone)
        db.add(cust)
        db.flush()
    f.seats -= len(passengers)
    b = Booking(flight_id=f.id, customer_id=cust.id, num_seats=len(passengers),
                total_price=f.price * len(passengers))
    b.passengers = [Passenger(**p) for p in passengers]
    db.add(b)
    db.commit()
    return booking_dict(b)


def get_booking(db: Session, booking_id: int) -> Booking:
    b = db.get(Booking, booking_id)
    if not b:
        raise BookingError(f"Booking {booking_id} not found", 404)
    return b


def refund_quote(b: Booking) -> dict:
    hours = (b.flight.departure - datetime.now()).total_seconds() / 3600
    if b.status == "CANCELLED":
        return dict(refund=0, rule="Already cancelled")
    if hours > 72:
        return dict(refund=max(b.total_price - CANCEL_FEE, 0), rule=f"More than 72h before departure: INR {CANCEL_FEE} fee")
    if hours >= 24:
        return dict(refund=b.total_price * 0.5, rule="24-72h before departure: 50% refund")
    return dict(refund=0, rule="Less than 24h before departure: no refund")


def cancel_booking(db: Session, booking_id: int) -> dict:
    b = get_booking(db, booking_id)
    if b.status == "CANCELLED":
        raise BookingError("Booking is already cancelled")
    quote = refund_quote(b)
    b.status = "CANCELLED"
    b.flight.seats += b.num_seats
    db.commit()
    return dict(booking=booking_dict(b), **quote)


def update_passengers(db: Session, booking_id: int, passengers: list[dict]) -> dict:
    b = get_booking(db, booking_id)
    if b.status != "CONFIRMED":
        raise BookingError("Only confirmed bookings can be updated")
    passengers = _clean_passengers(passengers)
    if len(passengers) != b.num_seats:
        raise BookingError(f"This booking has {b.num_seats} passenger(s); the count cannot change")
    b.passengers = [Passenger(**p) for p in passengers]
    db.commit()
    return booking_dict(b)

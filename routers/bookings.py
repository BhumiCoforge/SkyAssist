from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

import services as svc
from db import get_db
from schemas import BookingIn, BookingOut, CancelOut, PassengersUpdate, RefundQuote

router = APIRouter(prefix="/bookings", tags=["Bookings"])

# Every booking lookup needs the booking id AND the email it was booked with (like an airline PNR +
# surname check). This stops anyone from reading or cancelling other people's bookings by guessing ids.
Email = Query(..., min_length=5, max_length=120, description="Email used when booking")


@router.post("", status_code=201, response_model=BookingOut)
def create_booking(body: BookingIn, db: Session = Depends(get_db)):
    c = body.customer
    return svc.create_booking(db, body.flight_id, c.name, c.email, c.phone,
                              [p.model_dump() for p in body.passengers])


@router.get("/{booking_id}", response_model=BookingOut)
def get_booking(booking_id: int, email: str = Email, db: Session = Depends(get_db)):
    return svc.booking_dict(svc.get_owned_booking(db, booking_id, email))


@router.put("/{booking_id}", response_model=BookingOut)
def update_booking(booking_id: int, body: PassengersUpdate, email: str = Email,
                   db: Session = Depends(get_db)):
    return svc.update_passengers(db, booking_id, email, [p.model_dump() for p in body.passengers])


@router.get("/{booking_id}/cancel-quote", response_model=RefundQuote)
def cancel_quote(booking_id: int, email: str = Email, db: Session = Depends(get_db)):
    return svc.refund_quote(svc.get_owned_booking(db, booking_id, email))


@router.delete("/{booking_id}", response_model=CancelOut)
def cancel_booking(booking_id: int, email: str = Email, db: Session = Depends(get_db)):
    return svc.cancel_booking(db, booking_id, email)
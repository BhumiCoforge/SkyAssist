from fastapi import APIRouter, Depends, Query, Header
from sqlalchemy.orm import Session

import services as svc
import payments as pay
from db import get_db
from schemas import BookingIn, BookingOut, CancelOut, PassengersUpdate, RefundQuote, PaymentVerifyIn, PaymentFailureIn, DemoPaymentIn

router = APIRouter(prefix="/bookings", tags=["Bookings"])

# Every booking lookup needs the booking id AND the email it was booked with (like an airline PNR +
# surname check). This stops anyone from reading or cancelling other people's bookings by guessing ids.
Email = Query(..., min_length=5, max_length=120, description="Email used when booking")


@router.post("", status_code=201)
def create_booking(body: BookingIn, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    c = body.customer
    return pay.create_booking_order(body.flight_id, c.model_dump(),
                                   [p.model_dump() for p in body.passengers], idempotency_key)


@router.post("/{booking_id}/payment/verify")
def verify_booking_payment(booking_id: int, body: PaymentVerifyIn):
    return pay.verify_payment(booking_id, str(body.email), body.razorpay_order_id,
                             body.razorpay_payment_id, body.razorpay_signature)






@router.post("/{booking_id}/payment/demo")
def process_demo_booking_payment(booking_id: int, body: DemoPaymentIn):
    return pay.process_demo_payment(
        booking_id, str(body.email), body.payment_order_id, body.outcome
    )

@router.post("/{booking_id}/payment/failure")
def record_booking_payment_failure(booking_id: int, body: PaymentFailureIn):
    return pay.record_payment_failure(booking_id, str(body.email), body.razorpay_payment_id,
                                     body.error_code, body.error_description)

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


@router.delete("/{booking_id}")
def cancel_booking(booking_id: int, email: str = Email):
    return pay.cancel_paid_booking(booking_id, email)

"""Pydantic request and response models."""
from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, EmailStr, Field, StringConstraints

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=80)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=40)]


# ---------- requests ----------
class FlightIn(BaseModel):
    flight_no: Text
    airline: Text
    origin: Text
    destination: Text
    departure: datetime
    arrival: datetime
    price: float = Field(gt=0)
    seats: int = Field(default=50, ge=0, le=900)


class PassengerIn(BaseModel):
    name: Name
    age: Optional[int] = Field(default=None, ge=0, le=120)


class CustomerIn(BaseModel):
    name: Name
    email: EmailStr
    phone: str = Field(default="", max_length=20, pattern=r"^[0-9+\-\s()]*$")


class BookingIn(BaseModel):
    flight_id: int = Field(ge=1)
    customer: CustomerIn
    passengers: list[PassengerIn] = Field(min_length=1, max_length=9)


class PassengersUpdate(BaseModel):
    passengers: list[PassengerIn] = Field(min_length=1, max_length=9)


class ChatIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    message: str = Field(min_length=1, max_length=2000)


class SessionIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


# ---------- responses ----------
class FlightOut(BaseModel):
    id: int
    flight_no: str
    airline: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    price: float
    seats: int


class PassengerOut(BaseModel):
    name: str
    age: Optional[int] = None


class CustomerOut(BaseModel):
    name: str
    email: str
    phone: str = ""


class BookingOut(BaseModel):
    id: int
    status: str
    num_seats: int
    total_price: float
    created_at: datetime
    flight: FlightOut
    customer: CustomerOut
    passengers: list[PassengerOut]


class RefundQuote(BaseModel):
    refund: float
    rule: str


class CancelOut(RefundQuote):
    booking: BookingOut


class MessageOut(BaseModel):
    message: str

class PaymentVerifyIn(BaseModel):
    email: EmailStr
    razorpay_order_id: str = Field(min_length=5, max_length=100)
    razorpay_payment_id: str = Field(min_length=5, max_length=100)
    razorpay_signature: str = Field(min_length=20, max_length=200)


class PaymentVerifyOut(BaseModel):
    ok: bool
    booking: BookingOut
    payment_status: str


class PaymentFailureIn(BaseModel):
    email: EmailStr
    razorpay_payment_id: Optional[str] = Field(default=None, max_length=100)
    error_code: Optional[str] = Field(default=None, max_length=80)
    error_description: Optional[str] = Field(default=None, max_length=300)


class DemoPaymentIn(BaseModel):
    email: EmailStr
    payment_order_id: str = Field(min_length=5, max_length=100)
    outcome: str = Field(pattern=r"^(success|failure)$")

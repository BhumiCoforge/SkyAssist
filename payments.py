"""Razorpay payment gateway integration. Uses test credentials configured server-side only."""
import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta

from config import settings
from db import Booking, BookingStatus, Customer, Flight, Passenger, Payment, Refund, SessionLocal
import services as svc

log = logging.getLogger("skyassist.payments")


class PaymentError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.message, self.status = message, status


def _client():
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise PaymentError("Payment gateway is not configured. Add Razorpay TEST credentials to .env.", 503)
    if not settings.razorpay_key_id.startswith("rzp_test_"):
        raise PaymentError("Only Razorpay TEST keys are permitted in this application.", 403)
    try:
        import razorpay
    except ImportError:
        raise PaymentError("Razorpay SDK is missing. Install requirements.txt.", 503)
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


def create_booking_order(flight_id: int, customer: dict, passengers: list[dict], idempotency_key: str | None = None) -> dict:
    """Reserve seats, create a pending booking, then create a gateway order."""
    expire_stale_bookings()
    passengers = svc._clean_passengers(passengers)
    idempotency_key = idempotency_key or str(uuid.uuid4())
    if len(idempotency_key) > 128:
        raise PaymentError("Idempotency key is too long.", 422)
    with SessionLocal() as db:
        prior = db.query(Booking).filter_by(idempotency_key=idempotency_key).first()
        if prior:
            prior_payment = db.query(Payment).filter_by(booking_id=prior.id).first()
            if prior.status == BookingStatus.PENDING_PAYMENT.value and prior_payment:
                return {"booking_id": prior.id, "booking_status": prior.status,
                        "payment_order_id": prior_payment.order_id, "amount": prior.total_price,
                        "amount_paise": int(round(prior.total_price*100)), "currency": prior_payment.currency,
                        "key_id": settings.razorpay_key_id if prior_payment.gateway == "razorpay" else None,
                        "provider": prior_payment.gateway,
                        "expires_at": prior.payment_expires_at.isoformat() if prior.payment_expires_at else None,
                        "customer": {"name": prior.customer.name, "email": prior.customer.email, "phone": prior.customer.phone}}
            raise PaymentError("This idempotency key was already used. Start a new booking attempt.", 409)
        f = svc.check_bookable(db, flight_id, len(passengers))
        email = svc.normalize_email(customer["email"])
        cust = db.query(Customer).filter_by(email=email).first()
        if cust is None:
            cust = Customer(name=customer["name"].strip(), email=email, phone=customer.get("phone", ""))
            db.add(cust); db.flush()
        elif customer.get("phone") and not cust.phone:
            cust.phone = customer["phone"]
        reserve = db.query(Flight).filter(Flight.id == f.id, Flight.seats >= len(passengers)).update(
            {Flight.seats: Flight.seats - len(passengers)}, synchronize_session=False)
        if reserve != 1:
            db.rollback()
            raise PaymentError("Seats were just taken. Please search again.", 409)
        total = round(float(f.price) * len(passengers), 2)
        expires = datetime.now() + timedelta(minutes=settings.payment_order_ttl_minutes)
        b = Booking(flight_id=f.id, customer_id=cust.id, num_seats=len(passengers),
                    total_price=total, status=BookingStatus.PENDING_PAYMENT.value,
                    payment_expires_at=expires, idempotency_key=idempotency_key)
        b.passengers = [Passenger(**p) for p in passengers]
        db.add(b); db.commit(); db.refresh(b)
        booking_id = b.id
        db.refresh(f)
        flight_no = f.flight_no

    if settings.payment_provider.strip().lower() == "demo":
        demo_order_id = "demo_order_" + uuid.uuid4().hex[:16]
        with SessionLocal() as db:
            demo_payment = Payment(
                booking_id=booking_id,
                gateway="demo",
                order_id=demo_order_id,
                amount=total,
                currency="INR",
                status="CREATED",
            )
            db.add(demo_payment)
            db.commit()
        return {
            "booking_id": booking_id,
            "booking_status": "PENDING_PAYMENT",
            "payment_order_id": demo_order_id,
            "amount": total,
            "amount_paise": int(round(total * 100)),
            "currency": "INR",
            "key_id": None,
            "provider": "demo",
            "expires_at": expires.isoformat(),
            "customer": customer,
        }

    try:
        client = _client()
        order = client.order.create(data={"amount": int(round(total * 100)), "currency": "INR",
                                         "receipt": f"skyassist_booking_{booking_id}",
                                         "notes": {"booking_id": str(booking_id), "flight_no": flight_no}})
        with SessionLocal() as db:
            pay = Payment(booking_id=booking_id, order_id=order["id"], amount=total,
                         currency="INR", status="CREATED")
            db.add(pay); db.commit()
        return {"booking_id": booking_id, "booking_status": "PENDING_PAYMENT",
                "payment_order_id": order["id"], "amount": total, "amount_paise": int(round(total*100)),
                "currency": "INR", "key_id": settings.razorpay_key_id, "expires_at": expires.isoformat(),
                "customer": customer}
    except Exception as exc:
        # Gateway order failure: release the reserved seats and mark the booking cancelled.
        with SessionLocal() as db:
            b = db.get(Booking, booking_id)
            if b and b.status == BookingStatus.PENDING_PAYMENT.value:
                b.status = BookingStatus.CANCELLED.value
                db.execute(__import__("sqlalchemy").update(Flight).where(Flight.id == b.flight_id)
                           .values(seats=Flight.seats + b.num_seats))
                db.commit()
        if isinstance(exc, PaymentError):
            raise
        log.exception("Razorpay order creation failed")
        raise PaymentError("Unable to create payment order. No booking was confirmed.", 502)


def verify_payment(booking_id: int, email: str, order_id: str, payment_id: str, signature: str) -> dict:
    with SessionLocal() as db:
        b = svc.get_owned_booking(db, booking_id, email)
        pay = db.query(Payment).filter_by(booking_id=booking_id, order_id=order_id).first()
        if not pay:
            raise PaymentError("Payment order not found for this booking.", 404)
        if b.status == BookingStatus.CONFIRMED.value and pay.status == "PAID":
            if pay.payment_id == payment_id:
                return {"ok": True, "booking": svc.booking_dict(b), "payment_status": "PAID"}
            raise PaymentError("Booking is already paid with a different payment.", 409)
        if b.status != BookingStatus.PENDING_PAYMENT.value:
            raise PaymentError("This booking is not awaiting payment.", 409)
        if b.payment_expires_at and datetime.now() > b.payment_expires_at:
            expire_pending_booking(db, b)
            raise PaymentError("Payment session expired. Please create a new booking.", 410)
        secret = settings.razorpay_key_secret
        expected = hmac.new(secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature or ""):
            pay.status = "VERIFICATION_FAILED"
            db.commit()
            raise PaymentError("Payment signature verification failed.", 400)
        # Fetch payment from gateway to verify amount/order/status rather than trust browser callback.
        try:
            client = _client()
            gateway_payment = client.payment.fetch(payment_id)
            if gateway_payment.get("order_id") != order_id or int(gateway_payment.get("amount", -1)) != int(round(pay.amount*100)):
                raise PaymentError("Gateway payment details do not match this booking.", 400)
            if gateway_payment.get("currency") != pay.currency:
                raise PaymentError("Gateway payment currency does not match this booking.", 400)
            if gateway_payment.get("status") not in ("captured", "authorized"):
                raise PaymentError("Payment has not been captured/authorized.", 409)
        except PaymentError:
            raise
        except Exception:
            log.exception("Gateway payment lookup failed")
            raise PaymentError("Unable to verify payment status with gateway. Try again.", 502)
        # Atomic state transition prevents concurrent callbacks from confirming twice.
        from sqlalchemy import update
        claimed = db.execute(update(Booking).where(
            Booking.id == booking_id, Booking.status == BookingStatus.PENDING_PAYMENT.value
        ).values(status=BookingStatus.CONFIRMED.value, payment_expires_at=None))
        if claimed.rowcount != 1:
            db.rollback()
            current = svc.get_owned_booking(db, booking_id, email)
            current_payment = db.query(Payment).filter_by(booking_id=booking_id, order_id=order_id).first()
            if current.status == BookingStatus.CONFIRMED.value and current_payment and current_payment.payment_id == payment_id:
                return {"ok": True, "booking": svc.booking_dict(current), "payment_status": "PAID"}
            raise PaymentError("Booking payment state changed. Refresh and check booking status.", 409)
        pay.payment_id = payment_id
        pay.signature = signature
        pay.status = "PAID"
        db.commit()
        db.refresh(b)
        return {"ok": True, "booking": svc.booking_dict(b), "payment_status": pay.status}





def process_demo_payment(booking_id: int, email: str, order_id: str, outcome: str) -> dict:
    """Process a simulated payment locally. Never contacts an external gateway."""
    from sqlalchemy import update

    with SessionLocal() as db:
        booking = svc.get_owned_booking(db, booking_id, email)
        payment = db.query(Payment).filter_by(
            booking_id=booking_id, order_id=order_id, gateway="demo"
        ).first()

        if not payment:
            raise PaymentError("Demo payment order does not belong to this booking.", 404)

        # Repeated successful request returns the same confirmed booking.
        if booking.status == BookingStatus.CONFIRMED.value and payment.status == "PAID":
            return {
                "ok": True,
                "booking": svc.booking_dict(booking),
                "payment_status": "PAID",
                "demo": True,
            }

        if booking.status != BookingStatus.PENDING_PAYMENT.value:
            raise PaymentError("Booking is not awaiting payment.", 409)

        if booking.payment_expires_at and datetime.now() > booking.payment_expires_at:
            expire_pending_booking(db, booking)
            raise PaymentError("Payment session expired. Please create a new booking.", 410)

        if outcome not in ("success", "failure"):
            raise PaymentError("Invalid demo payment outcome.", 422)

        if outcome == "failure":
            payment.status = "FAILED"
            db.commit()
            return {
                "ok": False,
                "booking_status": booking.status,
                "payment_status": "FAILED",
                "message": "Simulated payment failed. You can retry payment.",
                "demo": True,
            }

        # Atomic transition prevents multiple successful payments confirming twice.
        claimed = db.execute(
            update(Booking)
            .where(
                Booking.id == booking_id,
                Booking.status == BookingStatus.PENDING_PAYMENT.value,
            )
            .values(status=BookingStatus.CONFIRMED.value, payment_expires_at=None)
        )
        if claimed.rowcount != 1:
            db.rollback()
            current = svc.get_owned_booking(db, booking_id, email)
            current_payment = db.query(Payment).filter_by(
                booking_id=booking_id, order_id=order_id, gateway="demo"
            ).first()
            if current.status == BookingStatus.CONFIRMED.value and current_payment and current_payment.status == "PAID":
                return {
                    "ok": True,
                    "booking": svc.booking_dict(current),
                    "payment_status": "PAID",
                    "demo": True,
                }
            raise PaymentError("Booking state changed. Refresh and check your booking.", 409)

        payment.payment_id = "demo_pay_" + uuid.uuid4().hex[:18]
        payment.status = "PAID"
        payment.signature = None
        db.commit()
        db.refresh(booking)
        return {
            "ok": True,
            "booking": svc.booking_dict(booking),
            "payment_status": "PAID",
            "demo": True,
        }

def record_payment_failure(booking_id: int, email: str, payment_id: str | None = None,
                          error_code: str | None = None, error_description: str | None = None) -> dict:
    with SessionLocal() as db:
        booking = svc.get_owned_booking(db, booking_id, email)
        if booking.status != BookingStatus.PENDING_PAYMENT.value:
            raise PaymentError("Booking is not awaiting payment.", 409)
        payment = db.query(Payment).filter_by(booking_id=booking_id).order_by(Payment.id.desc()).first()
        if not payment:
            raise PaymentError("Payment record not found.", 404)
        payment.status = "FAILED"
        db.commit()
        log.info("payment failed booking=%s code=%s", booking_id, error_code or "unknown")
        return {"ok": True, "payment_status": "FAILED", "booking_status": booking.status}


def expire_pending_booking(db, booking: Booking) -> None:
    if booking.status != BookingStatus.PENDING_PAYMENT.value:
        return
    booking.status = BookingStatus.CANCELLED.value
    db.execute(__import__("sqlalchemy").update(Flight).where(Flight.id == booking.flight_id)
               .values(seats=Flight.seats + booking.num_seats))
    payments = db.query(Payment).filter_by(booking_id=booking.id).all()
    for p in payments:
        if p.status == "CREATED":
            p.status = "EXPIRED"
    db.commit()


def expire_stale_bookings():
    with SessionLocal() as db:
        stale = db.query(Booking).filter(Booking.status == BookingStatus.PENDING_PAYMENT.value,
                                        Booking.payment_expires_at < datetime.now()).all()
        for b in stale:
            expire_pending_booking(db, b)


def cancel_paid_booking(booking_id: int, email: str) -> dict:
    """Claim cancellation atomically, request refund, then release seats; revert claim if refund fails."""
    from sqlalchemy import update
    with SessionLocal() as db:
        b = svc.get_owned_booking(db, booking_id, email)
        if b.status == BookingStatus.CANCELLED.value:
            raise svc.BookingError("Booking is already cancelled")
        if b.status != BookingStatus.CONFIRMED.value:
            raise svc.BookingError("Only confirmed bookings can be cancelled")
        if b.flight.departure <= datetime.now():
            raise svc.BookingError("This flight has already departed and cannot be cancelled")
        quote = svc.refund_quote(b)
        paid_record = db.query(Payment).filter_by(booking_id=booking_id, status="PAID").first()
        if not paid_record:
            quote = {"refund": 0, "rule": "No verified payment record; no gateway refund is due"}
        claimed = db.execute(update(Booking).where(
            Booking.id == booking_id, Booking.status == BookingStatus.CONFIRMED.value
        ).values(status=BookingStatus.CANCEL_PENDING.value))
        if claimed.rowcount != 1:
            db.rollback()
            raise svc.BookingError("Cancellation is already being processed.", 409)
        db.commit()
    refund_result = {"refund": 0, "refund_status": "NOT_ELIGIBLE"}
    try:
        if quote["refund"] > 0:
            refund_result = process_refund(booking_id, email, quote["refund"])
    except PaymentError:
        with SessionLocal() as db:
            db.execute(update(Booking).where(Booking.id == booking_id,
                                           Booking.status == BookingStatus.CANCEL_PENDING.value)
                       .values(status=BookingStatus.CONFIRMED.value))
            db.commit()
        raise
    with SessionLocal() as db:
        b = svc.get_owned_booking(db, booking_id, email)
        final = db.execute(update(Booking).where(Booking.id == booking_id,
                                               Booking.status == BookingStatus.CANCEL_PENDING.value)
                           .values(status=BookingStatus.CANCELLED.value))
        if final.rowcount != 1:
            db.rollback()
            raise svc.BookingError("Cancellation state changed; check your booking status.", 409)
        db.execute(update(Flight).where(Flight.id == b.flight_id).values(seats=Flight.seats + b.num_seats))
        db.commit()
        db.refresh(b)
        db.refresh(b.flight)
        return {"booking": svc.booking_dict(b), **quote, "refund_status": refund_result.get("refund_status"),
                "refund_id": refund_result.get("refund_id")}


def process_refund(booking_id: int, email: str, amount: float) -> dict:
    with SessionLocal() as db:
        b = svc.get_owned_booking(db, booking_id, email)
        paid = db.query(Payment).filter_by(booking_id=booking_id, status="PAID").first()
        if not paid:
            return {"refund": 0, "refund_status": "NOT_PAID"}
        if amount <= 0:
            return {"refund": 0, "refund_status": "NOT_ELIGIBLE"}
        if amount > paid.amount:
            raise PaymentError("Refund amount exceeds the paid amount.", 400)
        prior = db.query(Refund).filter(Refund.payment_id == paid.id,
                                        Refund.status.in_(("REQUESTED", "PENDING", "PROCESSED"))).first()
        if prior:
            raise PaymentError("A refund has already been requested for this payment.", 409)
        refund = Refund(payment_id=paid.id, amount=amount, status="REQUESTED")
        db.add(refund)
        db.commit()

        if paid.gateway == "demo":
            refund.gateway_refund_id = "demo_refund_" + uuid.uuid4().hex[:16]
            refund.status = "PROCESSED"
            db.commit()
            return {
                "refund": amount,
                "refund_status": "PROCESSED",
                "refund_id": refund.gateway_refund_id,
            }

        try:
            client = _client()
            gateway_refund = client.payment.refund(paid.payment_id, {"amount": int(round(amount*100))})
            refund.gateway_refund_id = gateway_refund.get("id")
            refund.status = gateway_refund.get("status", "REQUESTED").upper()
            db.commit()
            return {"refund": amount, "refund_status": refund.status, "refund_id": refund.gateway_refund_id}
        except PaymentError:
            raise
        except Exception:
            refund.status = "FAILED"
            db.commit()
            log.exception("Refund request failed")
            raise PaymentError("Refund request failed at the gateway. Booking cancellation was not completed.", 502)

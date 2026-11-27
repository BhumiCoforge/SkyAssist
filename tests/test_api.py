import hashlib, hmac
from fastapi.testclient import TestClient
import payments
from config import settings
import services as svc

from main import app
from tests.conftest import make_flight

client = TestClient(app)   # no `with`: skips startup seeding, tests create their own flights
ADMIN = {"X-Admin-Key": "test-admin-key"}
class FakeGateway:
    def __init__(self):
        self.order = self
        self.payment = self
        self.refunds = []
    def create(self, data):
        self.last_order = data
        return {"id": "order_test_12345"}
    def fetch(self, payment_id):
        return {"id": payment_id, "order_id": "order_test_12345", "amount": self.last_order["amount"],
                "currency": "INR", "status": "captured"}
    def refund(self, payment_id, data):
        self.refunds.append((payment_id, data))
        return {"id": "rfnd_test_123", "status": "processed"}


def setup_gateway(monkeypatch):
    monkeypatch.setattr(settings, "payment_provider", "razorpay")
    gateway = FakeGateway()
    monkeypatch.setattr(settings, "razorpay_key_id", "rzp_test_key")
    monkeypatch.setattr(settings, "razorpay_key_secret", "test_secret")
    monkeypatch.setattr(payments, "_client", lambda: gateway)
    return gateway



def body(flight_id, email="asha@example.com", n=1):
    return dict(flight_id=flight_id, customer=dict(name="Asha Rao", email=email, phone="+91 98765 43210"),
                passengers=[dict(name=f"Pax Number{i}", age=30) for i in range(n)])


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_search_validates_query_params(flight):
    assert client.get("/flights", params={"origin": "Delhi", "destination": "Mumbai"}).status_code == 200
    assert client.get("/flights", params={"date": "not-a-date"}).status_code == 422
    assert client.get("/flights", params={"passengers": 0}).status_code == 422
    assert client.get("/flights", params={"limit": 1000}).status_code == 422


def test_full_booking_flow(flight, monkeypatch):
    gateway = setup_gateway(monkeypatch)
    r = client.post("/bookings", json=body(flight.id, n=2))
    assert r.status_code == 201, r.text
    order = r.json()
    bid = order["booking_id"]
    assert order["amount"] == 10000
    assert client.get(f"/bookings/{bid}", params={"email": "asha@example.com"}).json()["status"] == "PENDING_PAYMENT"
    sig = hmac.new(b"test_secret", b"order_test_12345|pay_test_12345", hashlib.sha256).hexdigest()
    verified = client.post(f"/bookings/{bid}/payment/verify", json={
        "email": "asha@example.com", "razorpay_order_id": "order_test_12345",
        "razorpay_payment_id": "pay_test_12345", "razorpay_signature": sig})
    assert verified.status_code == 200, verified.text
    assert verified.json()["booking"]["status"] == "CONFIRMED"
    q = client.get(f"/bookings/{bid}/cancel-quote", params={"email": "asha@example.com"}).json()
    assert q["refund"] == 10000 - 500
    r = client.delete(f"/bookings/{bid}", params={"email": "asha@example.com"})
    assert r.status_code == 200 and r.json()["booking"]["status"] == "CANCELLED"
    assert r.json()["refund_status"] == "PROCESSED"
    assert gateway.refunds
    assert client.delete(f"/bookings/{bid}", params={"email": "asha@example.com"}).status_code == 400


def test_booking_needs_email(flight, monkeypatch):
    setup_gateway(monkeypatch)
    bid = client.post("/bookings", json=body(flight.id)).json()["booking_id"]
    assert client.get(f"/bookings/{bid}").status_code == 422                                   # email missing
    assert client.get(f"/bookings/{bid}", params={"email": "x@y.com"}).status_code == 404       # wrong email
    assert client.delete(f"/bookings/{bid}", params={"email": "x@y.com"}).status_code == 404
    assert client.put(f"/bookings/{bid}", params={"email": "x@y.com"},
                      json={"passengers": [dict(name="Hacker Name")]}).status_code == 404


def test_booking_validation(flight):
    bad = body(flight.id)
    bad["customer"]["email"] = "not-an-email"
    assert client.post("/bookings", json=bad).status_code == 422
    bad = body(flight.id)
    bad["passengers"] = []
    assert client.post("/bookings", json=bad).status_code == 422
    bad = body(flight.id)
    bad["passengers"][0]["name"] = "A"
    assert client.post("/bookings", json=bad).status_code == 422
    assert client.post("/bookings", json=body(9999)).status_code == 404
    assert client.post("/bookings", json=body(flight.id, n=11)).status_code == 422


def test_overbooking_rejected(db):
    f = make_flight(db, seats=1)
    assert client.post("/bookings", json=body(f.id, n=2)).status_code == 400


def test_flight_admin_endpoints_need_key(flight):
    new = dict(flight_no="XY999", airline="SkyAir", origin="Delhi", destination="Goa",
               departure="2030-01-01T10:00:00", arrival="2030-01-01T12:00:00", price=4000, seats=20)
    assert client.post("/flights", json=new).status_code == 401
    assert client.post("/flights", json=new, headers={"X-Admin-Key": "wrong"}).status_code == 401
    r = client.post("/flights", json=new, headers=ADMIN)
    assert r.status_code == 201
    fid = r.json()["id"]
    assert client.delete(f"/flights/{fid}").status_code == 401
    assert client.delete(f"/flights/{fid}", headers=ADMIN).status_code == 200
    bad = {**new, "arrival": "2029-01-01T12:00:00"}
    assert client.post("/flights", json=bad, headers=ADMIN).status_code == 422


def test_cannot_delete_flight_with_bookings(flight, db):
    svc.create_booking(db, flight.id, "Asha Rao", "asha@example.com", "", [dict(name="Passenger Name")])
    assert client.delete(f"/flights/{flight.id}", headers=ADMIN).status_code == 409


def test_rag_rebuild_needs_admin():
    assert client.post("/rag/rebuild").status_code == 401


def test_chat_rejects_bad_session_id():
    assert client.post("/chat", json=dict(session_id="x", message="hi")).status_code == 422
    assert client.post("/chat", json=dict(session_id="../../etc/passwd", message="hi")).status_code == 422

def test_invalid_signature_does_not_confirm_and_success_is_idempotent(flight, monkeypatch):
    setup_gateway(monkeypatch)
    order = client.post("/bookings", json=body(flight.id)).json()
    bid = order["booking_id"]
    invalid = client.post(f"/bookings/{bid}/payment/verify", json={
        "email": "asha@example.com", "razorpay_order_id": order["payment_order_id"],
        "razorpay_payment_id": "pay_test_12345", "razorpay_signature": "x" * 30})
    assert invalid.status_code == 400
    assert client.get(f"/bookings/{bid}", params={"email": "asha@example.com"}).json()["status"] == "PENDING_PAYMENT"
    sig = hmac.new(b"test_secret", f"{order['payment_order_id']}|pay_test_12345".encode(), hashlib.sha256).hexdigest()
    payload = {"email": "asha@example.com", "razorpay_order_id": order["payment_order_id"],
               "razorpay_payment_id": "pay_test_12345", "razorpay_signature": sig}
    success = client.post(f"/bookings/{bid}/payment/verify", json=payload)
    assert success.status_code == 200
    repeat = client.post(f"/bookings/{bid}/payment/verify", json=payload)
    assert repeat.status_code == 200 and repeat.json()["payment_status"] == "PAID"

def test_refund_gateway_failure_does_not_cancel_booking(flight, monkeypatch):
    gateway = setup_gateway(monkeypatch)
    order = client.post("/bookings", json=body(flight.id)).json()
    bid = order["booking_id"]
    sig = hmac.new(b"test_secret", f"{order['payment_order_id']}|pay_test_12345".encode(), hashlib.sha256).hexdigest()
    verified = client.post(f"/bookings/{bid}/payment/verify", json={
        "email": "asha@example.com", "razorpay_order_id": order["payment_order_id"],
        "razorpay_payment_id": "pay_test_12345", "razorpay_signature": sig})
    assert verified.status_code == 200
    def fail_refund(payment_id, data):
        raise RuntimeError("simulated gateway refund failure")
    gateway.refund = fail_refund
    cancel = client.delete(f"/bookings/{bid}", params={"email": "asha@example.com"})
    assert cancel.status_code == 502
    booking = client.get(f"/bookings/{bid}", params={"email": "asha@example.com"}).json()
    assert booking["status"] == "CONFIRMED"

def test_booking_order_idempotency_key_prevents_duplicate_booking(flight, monkeypatch):
    setup_gateway(monkeypatch)
    payload = body(flight.id)
    headers = {"Idempotency-Key": "retry-safe-booking-key"}
    first = client.post("/bookings", json=payload, headers=headers)
    second = client.post("/bookings", json=payload, headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["booking_id"] == second.json()["booking_id"]
    # Only one seat should be reserved for the repeated request.
    from db import Flight, SessionLocal
    with SessionLocal() as db:
        current = db.get(Flight, flight.id)
        assert current.seats == flight.seats - 1

def test_payment_failure_is_recorded_but_retry_can_succeed(flight, monkeypatch):
    setup_gateway(monkeypatch)
    order = client.post("/bookings", json=body(flight.id)).json()
    bid = order["booking_id"]
    failed = client.post(f"/bookings/{bid}/payment/failure", json={
        "email": "asha@example.com", "razorpay_payment_id": "pay_failed_123",
        "error_code": "BAD_REQUEST_ERROR", "error_description": "Payment declined"})
    assert failed.status_code == 200 and failed.json()["payment_status"] == "FAILED"
    sig = hmac.new(b"test_secret", f"{order['payment_order_id']}|pay_test_12345".encode(), hashlib.sha256).hexdigest()
    retry = client.post(f"/bookings/{bid}/payment/verify", json={
        "email": "asha@example.com", "razorpay_order_id": order["payment_order_id"],
        "razorpay_payment_id": "pay_test_12345", "razorpay_signature": sig})
    assert retry.status_code == 200 and retry.json()["booking"]["status"] == "CONFIRMED"

def test_gateway_order_creation_failure_releases_reserved_seats(flight, monkeypatch):
    setup_gateway(monkeypatch)
    class BrokenGateway:
        order = None
        def __init__(self): self.order = self
        def create(self, data): raise RuntimeError("simulated order API failure")
    monkeypatch.setattr(payments, "_client", lambda: BrokenGateway())
    r = client.post("/bookings", json=body(flight.id))
    assert r.status_code == 502
    from db import Flight, SessionLocal, Booking
    with SessionLocal() as db:
        assert db.get(Flight, flight.id).seats == flight.seats
        assert db.query(Booking).filter_by(flight_id=flight.id).first().status == "CANCELLED"

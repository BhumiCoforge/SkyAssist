from fastapi.testclient import TestClient

from main import app
from tests.conftest import make_flight

client = TestClient(app)   # no `with`: skips startup seeding, tests create their own flights
ADMIN = {"X-Admin-Key": "test-admin-key"}


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


def test_full_booking_flow(flight):
    r = client.post("/bookings", json=body(flight.id, n=2))
    assert r.status_code == 201, r.text
    bid = r.json()["id"]
    assert client.get(f"/bookings/{bid}", params={"email": "asha@example.com"}).status_code == 200
    q = client.get(f"/bookings/{bid}/cancel-quote", params={"email": "asha@example.com"}).json()
    assert q["refund"] == 10000 - 500
    r = client.delete(f"/bookings/{bid}", params={"email": "asha@example.com"})
    assert r.status_code == 200 and r.json()["booking"]["status"] == "CANCELLED"
    assert client.delete(f"/bookings/{bid}", params={"email": "asha@example.com"}).status_code == 400


def test_booking_needs_email(flight):
    bid = client.post("/bookings", json=body(flight.id)).json()["id"]
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


def test_cannot_delete_flight_with_bookings(flight):
    client.post("/bookings", json=body(flight.id))
    assert client.delete(f"/flights/{flight.id}", headers=ADMIN).status_code == 409


def test_rag_rebuild_needs_admin():
    assert client.post("/rag/rebuild").status_code == 401


def test_chat_rejects_bad_session_id():
    assert client.post("/chat", json=dict(session_id="x", message="hi")).status_code == 422
    assert client.post("/chat", json=dict(session_id="../../etc/passwd", message="hi")).status_code == 422
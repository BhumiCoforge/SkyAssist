from datetime import datetime, timedelta

import pytest

import services as svc
from db import Booking, Customer, Flight, seed
from tests.conftest import PAX, make_flight


def book(db, f, email="asha@example.com", n=1, name="Asha Rao"):
    return svc.create_booking(db, f.id, name, email, "", [dict(name=f"P{i} Name") for i in range(n)])


def test_booking_reduces_seats_and_totals_price(db, flight):
    b = book(db, flight, n=2)
    assert b["total_price"] == 10000 and b["num_seats"] == 2
    assert db.get(Flight, flight.id).seats == 8


def test_cannot_book_more_seats_than_available(db):
    f = make_flight(db, seats=1)
    with pytest.raises(svc.BookingError, match="Only 1 seat"):
        book(db, f, n=2)


def test_cannot_book_departed_flight(db):
    f = make_flight(db, hours_ahead=-2)
    with pytest.raises(svc.BookingError, match="already departed"):
        book(db, f)


def test_needs_at_least_one_named_passenger(db, flight):
    with pytest.raises(svc.BookingError):
        svc.create_booking(db, flight.id, "A B", "a@b.com", "", [dict(name="  ")])


def test_existing_customer_is_not_overwritten(db, flight):
    book(db, flight, email="victim@example.com", name="Real Owner")
    book(db, flight, email="victim@example.com", name="Attacker")
    assert db.query(Customer).filter_by(email="victim@example.com").one().name == "Real Owner"


def test_booking_needs_matching_email(db, flight):
    b = book(db, flight)
    assert svc.get_owned_booking(db, b["id"], "ASHA@example.com ").id == b["id"]   # case/space tolerant
    with pytest.raises(svc.BookingError) as e:
        svc.get_owned_booking(db, b["id"], "someone@else.com")
    assert e.value.status == 404
    with pytest.raises(svc.BookingError):
        svc.get_owned_booking(db, 9999, "asha@example.com")   # same error as wrong email


def test_cancel_releases_seats_and_second_cancel_fails(db, flight):
    b = book(db, flight, n=2)
    r = svc.cancel_booking(db, b["id"], "asha@example.com")
    assert r["booking"]["status"] == "CANCELLED"
    assert db.get(Flight, flight.id).seats == 10
    with pytest.raises(svc.BookingError, match="already cancelled"):
        svc.cancel_booking(db, b["id"], "asha@example.com")
    assert db.get(Flight, flight.id).seats == 10   # not released twice


def test_cancel_with_wrong_email_changes_nothing(db, flight):
    b = book(db, flight)
    with pytest.raises(svc.BookingError):
        svc.cancel_booking(db, b["id"], "other@example.com")
    assert db.get(Booking, b["id"]).status == "CONFIRMED"


@pytest.mark.parametrize("hours,expected", [
    (73, 5000 - svc.CANCEL_FEE),   # more than 72h: fee only
    (72.5, 5000 - svc.CANCEL_FEE),
    (71, 2500),                    # 24-72h: 50%
    (24.5, 2500),
    (23, 0),                       # under 24h: nothing
])
def test_refund_rules(db, hours, expected):
    f = make_flight(db, hours_ahead=hours)
    b = book(db, f)
    assert svc.refund_quote(db.get(Booking, b["id"]))["refund"] == expected


def test_cannot_cancel_after_departure(db, flight):
    b = book(db, flight)
    flight_row = db.get(Flight, flight.id)
    flight_row.departure = datetime.now() - timedelta(hours=1)
    db.commit()
    with pytest.raises(svc.BookingError, match="already departed"):
        svc.cancel_booking(db, b["id"], "asha@example.com")


def test_update_passengers_keeps_count(db, flight):
    b = book(db, flight, n=2)
    with pytest.raises(svc.BookingError, match="count cannot change"):
        svc.update_passengers(db, b["id"], "asha@example.com", [dict(name="Only One")])
    out = svc.update_passengers(db, b["id"], "asha@example.com", [dict(name="New One"), dict(name="New Two")])
    assert [p["name"] for p in out["passengers"]] == ["New One", "New Two"]


def test_city_aliases_in_search(db):
    make_flight(db, origin="Bengaluru", destination="Mumbai")
    assert len(svc.search_flights(db, origin="Bangalore", destination="Bombay")) == 1
    assert len(svc.search_flights(db, origin="BLR", destination="BOM")) == 1


def test_search_filters_price_seats_and_pagination(db):
    for i in range(5):
        make_flight(db, price=4000 + i * 1000, seats=2, flight_no=f"SA10{i}", hours_ahead=50 + i)
    assert len(svc.search_flights(db, max_price=5000)) == 2
    assert svc.search_flights(db, passengers=3) == []
    assert len(svc.search_flights(db, limit=2)) == 2
    assert len(svc.search_flights(db, limit=2, offset=4)) == 1


def test_search_rejects_bad_date(db):
    with pytest.raises(svc.BookingError):
        svc.search_flights(db, date="20/12/2026")


def test_seed_tops_up_missing_days_only(db):
    first = seed(days=3)
    assert first > 0
    assert seed(days=3) == 0             # nothing missing
    assert seed(days=5) > 0              # adds only days 3-4
    days = {f.departure.date() for f in db.query(Flight)}
    assert len(days) == 5
import os
import tempfile
from datetime import datetime, timedelta

# Must be set before the app modules are imported: use a throw-away database.
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["ANTHROPIC_API_KEY"] = "test-not-a-real-key"

import pytest  # noqa: E402

from db import Base, Flight, SessionLocal, engine  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def make_flight(db, hours_ahead=100, seats=10, price=5000.0, origin="Delhi", destination="Mumbai",
                flight_no="SA101") -> Flight:
    dep = datetime.now() + timedelta(hours=hours_ahead)
    f = Flight(flight_no=flight_no, airline="SkyAir", origin=origin, destination=destination,
               departure=dep, arrival=dep + timedelta(hours=2), price=price, seats=seats)
    db.add(f)
    db.commit()
    return f


@pytest.fixture
def flight(db):
    return make_flight(db)


PAX = [dict(name="Asha Rao", age=30)]
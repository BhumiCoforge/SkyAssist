import enum
import random
from datetime import date, datetime, time, timedelta
from typing import Optional

from sqlalchemy import CheckConstraint, ForeignKey, Index, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from config import settings

DATABASE_URL = settings.database_url
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class BookingStatus(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class Base(DeclarativeBase):
    pass


class Flight(Base):
    __tablename__ = "flights"
    __table_args__ = (
        CheckConstraint("seats >= 0", name="ck_flight_seats_nonneg"),
        CheckConstraint("price > 0", name="ck_flight_price_pos"),
        Index("ix_flights_route_departure", "origin", "destination", "departure"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    flight_no: Mapped[str]
    airline: Mapped[str]
    origin: Mapped[str]
    destination: Mapped[str]
    departure: Mapped[datetime] = mapped_column(index=True)
    arrival: Mapped[datetime]
    price: Mapped[float]
    seats: Mapped[int] = mapped_column(default=50)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    phone: Mapped[str] = mapped_column(default="")


class Booking(Base):
    __tablename__ = "bookings"
    id: Mapped[int] = mapped_column(primary_key=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id"), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    num_seats: Mapped[int]
    total_price: Mapped[float]
    status: Mapped[str] = mapped_column(default=BookingStatus.CONFIRMED.value)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    flight: Mapped["Flight"] = relationship()
    customer: Mapped["Customer"] = relationship()
    passengers: Mapped[list["Passenger"]] = relationship(cascade="all, delete-orphan")


class Passenger(Base):
    __tablename__ = "passengers"
    id: Mapped[int] = mapped_column(primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id"))
    name: Mapped[str]
    age: Mapped[Optional[int]] = mapped_column(default=None)


class ChatMessage(Base):
    """Chat memory (Phase 4: persisted so a server restart does not wipe conversations)."""
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(index=True)
    payload: Mapped[str]          # one LangChain message as JSON
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class PendingAction(Base):
    """A booking/cancellation the AI proposed and the customer has not confirmed yet."""
    __tablename__ = "pending_actions"
    id: Mapped[str] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(index=True)
    action: Mapped[str]           # "book" | "cancel"
    summary: Mapped[str]
    payload: Mapped[str]          # JSON with the arguments to run on confirm
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# origin, destination, base price (INR), duration (minutes)
ROUTES = [
    ("Delhi", "Mumbai", 5200, 130), ("Delhi", "Bengaluru", 6500, 170),
    ("Delhi", "Lucknow", 3500, 75), ("Delhi", "Goa", 7200, 170),
    ("Delhi", "Kolkata", 5600, 135), ("Delhi", "Hyderabad", 5800, 150),
    ("Mumbai", "Goa", 3100, 70), ("Mumbai", "Bengaluru", 3900, 100),
    ("Mumbai", "Hyderabad", 4000, 95), ("Bengaluru", "Chennai", 2900, 65),
]
AIRLINES = [("SkyAir", "SA"), ("BlueJet", "BJ"), ("Sunwing", "SW")]
SLOTS = [(6, 30, 0.9), (11, 15, 1.0), (16, 45, 1.1), (20, 30, 1.25)]


def seed(days: Optional[int] = None) -> int:
    """Make sure sample flights exist for the next `days` days. Safe to call on every start:
    it only adds the missing days, so the site never runs out of flights. Returns rows added."""
    days = days or settings.seed_days
    with SessionLocal() as db:
        last = db.scalar(select(func.max(Flight.departure)))
        first_day = 0 if last is None else max((last.date() - date.today()).days + 1, 0)
        if first_day >= days:
            return 0
        rnd = random.Random(first_day)
        count = db.scalar(select(func.count(Flight.id))) or 0
        rows, n = [], count
        for o, d, base, mins in ROUTES:
            for a, b in ((o, d), (d, o)):
                for day in range(first_day, days):
                    for i, (h, m, mult) in enumerate(SLOTS):
                        n += 1
                        name, code = AIRLINES[(n + i) % 3]
                        dep = datetime.combine(date.today() + timedelta(days=day), time(h, m))
                        price = round(base * mult * rnd.uniform(0.92, 1.08) / 50) * 50
                        rows.append(Flight(
                            flight_no=f"{code}{100 + n % 900}", airline=name, origin=a,
                            destination=b, departure=dep, arrival=dep + timedelta(minutes=mins),
                            price=price, seats=rnd.randint(8, 60)))
        db.add_all(rows)
        db.commit()
        return len(rows)
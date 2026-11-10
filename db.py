import os
import random
from datetime import date, datetime, time, timedelta
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skyassist.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Flight(Base):
    __tablename__ = "flights"
    id: Mapped[int] = mapped_column(primary_key=True)
    flight_no: Mapped[str]
    airline: Mapped[str]
    origin: Mapped[str]
    destination: Mapped[str]
    departure: Mapped[datetime]
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
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id"))
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    num_seats: Mapped[int]
    total_price: Mapped[float]
    status: Mapped[str] = mapped_column(default="CONFIRMED")  # CONFIRMED | CANCELLED
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


def seed(days: int = 45):
    """Create sample flights for the next `days` days if the table is empty."""
    with SessionLocal() as db:
        if db.query(Flight).count():
            return
        rnd = random.Random(42)
        rows, n = [], 0
        for o, d, base, mins in ROUTES:
            for a, b in ((o, d), (d, o)):
                for day in range(days):
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

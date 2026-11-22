from datetime import date as Date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import services as svc
from db import Booking, Flight, get_db
from deps import require_admin
from schemas import FlightIn, FlightOut, MessageOut

router = APIRouter(prefix="/flights", tags=["Flights"])


@router.get("", response_model=list[FlightOut])
def search_flights(
    origin: str = Query("", max_length=40),
    destination: str = Query("", max_length=40),
    day: Optional[Date] = Query(None, alias="date", description="YYYY-MM-DD"),
    max_price: float = Query(0, ge=0),
    passengers: int = Query(1, ge=1, le=9),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return svc.search_flights(db, origin, destination, day.isoformat() if day else "",
                              max_price, passengers, limit, offset)


@router.get("/{flight_id}", response_model=FlightOut)
def get_flight(flight_id: int, db: Session = Depends(get_db)):
    f = db.get(Flight, flight_id)
    if not f:
        raise HTTPException(404, "Flight not found")
    return svc.flight_dict(f)


@router.post("", status_code=201, response_model=FlightOut, dependencies=[Depends(require_admin)])
def create_flight(flight: FlightIn, db: Session = Depends(get_db)):
    if flight.arrival <= flight.departure:
        raise HTTPException(422, "Arrival must be after departure")
    f = Flight(**flight.model_dump())
    db.add(f)
    db.commit()
    return svc.flight_dict(f)


@router.put("/{flight_id}", response_model=FlightOut, dependencies=[Depends(require_admin)])
def update_flight(flight_id: int, flight: FlightIn, db: Session = Depends(get_db)):
    f = db.get(Flight, flight_id)
    if not f:
        raise HTTPException(404, "Flight not found")
    if flight.arrival <= flight.departure:
        raise HTTPException(422, "Arrival must be after departure")
    for k, v in flight.model_dump().items():
        setattr(f, k, v)
    db.commit()
    return svc.flight_dict(f)


@router.delete("/{flight_id}", response_model=MessageOut, dependencies=[Depends(require_admin)])
def delete_flight(flight_id: int, db: Session = Depends(get_db)):
    f = db.get(Flight, flight_id)
    if not f:
        raise HTTPException(404, "Flight not found")
    if db.query(Booking).filter_by(flight_id=flight_id).first():
        raise HTTPException(409, "Flight has bookings and cannot be deleted")
    db.delete(f)
    db.commit()
    return {"message": "Flight deleted"}
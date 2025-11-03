# app/api/endpoints.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.db.database import get_db
from app.db import models
from . import schemas

router = APIRouter()

@router.get("/options-chain/{currency}", response_model=List[schemas.OptionInstrument])
def get_options_chain(
    currency: str,
    hours: int = Query(24, ge=1, le=168, description="Lookback window in hours"),
    db: Session = Depends(get_db),
):
    """
    Return recent normalized option instruments for `currency` (e.g., BTC, ETH)
    using the canonical shape expected by collectors/ETL/inference.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # OptionsChain.symbol stores the underlying symbol you used in collectors (typically 'BTC', not 'BTC/USDT')
    q = (
        db.query(models.OptionsChain)
        .filter(
            and_(
                models.OptionsChain.symbol == currency.upper(),
                models.OptionsChain.timestamp >= since,
            )
        )
        .order_by(models.OptionsChain.timestamp.desc())
        .limit(5000)
    )
    rows = q.all()
    if not rows:
        return []

    out: List[schemas.OptionInstrument] = []
    for r in rows:
        out.append(
            schemas.OptionInstrument(
                symbol=r.symbol,
                expiry=r.expiry,
                strike=r.strike,
                option_type=("CALL" if str(r.option_type).upper().startswith("C") else "PUT"),
                timestamp=r.timestamp,
                bid=r.bid,
                ask=r.ask,
                last_price=r.last_price,
                mark_price=r.mark_price,
                volume=r.volume,
                open_interest=r.open_interest,
                iv=r.iv,
                delta=r.delta,
                gamma=r.gamma,
                theta=r.theta,
                vega=r.vega,
            )
        )
    return out


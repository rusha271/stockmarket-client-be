"""Next NSE 5-min candle time (IST). Market 9:15–15:30, Mon–Fri."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE equity: market 9:15–15:30
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
CANDLE_MINUTES = 5

# Peak hours: morning open (9:15–10:00) and evening/close (15:00–15:30) – don't use small bins
PEAK_MORNING_END = time(10, 0)   # 9:15–10:00 = peak
PEAK_EVENING_START = time(15, 0)  # 15:00–15:30 = peak


def is_peak_hours_ist(now: datetime | None = None) -> bool:
    """True if current time (IST) is in morning or evening peak; then we do not apply small 5-min bins."""
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)
    t = now.time()
    if t < MARKET_OPEN or t >= MARKET_CLOSE:
        return False  # outside market, no peak
    # Morning peak: 9:15–10:00
    if MARKET_OPEN <= t < PEAK_MORNING_END:
        return True
    # Evening peak: 15:00–15:30
    if t >= PEAK_EVENING_START:
        return True
    return False


def get_next_5min_candle_ist(now: datetime | None = None) -> str:
    """
    Return human-readable timeframe for the next 5-min candle on NSE (IST).
    Only trading days (Mon–Fri); never Saturday or Sunday.
    E.g. "Next 5-min: Monday 09:20 IST". If now is None, use current time in IST.
    """
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    today = now.date()
    t = now.time()

    # Weekend (Saturday/Sunday): next candle is Monday 09:15
    if today.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return _format_timeframe(_next_trading_open(today))

    # Before market open today -> next candle is 9:15 today (today is Mon–Fri)
    if t < MARKET_OPEN:
        cand_ts = datetime.combine(today, MARKET_OPEN, tzinfo=IST)
        return _format_timeframe(cand_ts)

    # After market close -> next trading day 9:15
    if t >= MARKET_CLOSE:
        return _format_timeframe(_next_trading_open(today))

    # Within market hours: next 5-min slot (9:15, 9:20, ..., 15:25, 15:30)
    slot = (now.minute // CANDLE_MINUTES) * CANDLE_MINUTES
    next_min = slot + CANDLE_MINUTES
    if next_min >= 60:
        next_dt = datetime.combine(today, t.replace(minute=0, second=0, microsecond=0), tzinfo=IST) + timedelta(hours=1)
    else:
        next_dt = now.replace(minute=next_min, second=0, microsecond=0)
    if next_dt.time() > MARKET_CLOSE:
        return _format_timeframe(_next_trading_open(today))
    return _format_timeframe(next_dt)


def _next_trading_open(after_date=None):
    d = after_date if after_date is not None else datetime.now(IST).date()
    while True:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            return datetime.combine(d, MARKET_OPEN, tzinfo=IST)


def _format_timeframe(dt: datetime) -> str:
    """e.g. 'Next 5-min: Monday 09:20 IST' or 'Next 5-min: 15 Feb 2025 14:35 IST'."""
    name = dt.strftime("%A %H:%M IST")  # Monday 09:20 IST
    return f"Next 5-min: {name}"

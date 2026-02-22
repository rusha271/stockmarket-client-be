"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.api.routes import router
from app.api.auth_routes import router as auth_router
from app.db import init_auth_table

settings = Settings()


def _five_min_cron(app: FastAPI):
    clf = getattr(app.state, "clf", None)
    reg = getattr(app.state, "reg", None)
    if clf is not None and reg is not None:
        from app.predict_cron import run_five_min_job
        try:
            run_five_min_job(clf, reg)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Cron job failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    clf = xgb.Booster()
    clf.load_model(str(settings.classifier_model_path))
    reg = xgb.Booster()
    reg.load_model(str(settings.regression_model_path))
    app.state.clf = clf
    app.state.reg = reg
    try:
        init_auth_table()
    except Exception:
        pass  # DB may not be configured yet

    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    # Market: 9:15–15:30 IST Mon–Fri. Run at 9:15, 9:20, … 15:25, 15:30 (last run at 15:30).
    # Hours 9–14: every 5 min at :15,:20,:25,:30,:35,:40,:45,:50,:55
    scheduler.add_job(lambda: _five_min_cron(app), "cron", minute="15,20,25,30,35,40,45,50,55", hour="9,10,11,12,13,14", day_of_week="mon-fri")
    # Hour 15: only 15:15, 15:20, 15:25, 15:30 (stop at market close 15:30)
    scheduler.add_job(lambda: _five_min_cron(app), "cron", minute="15,20,25,30", hour="15", day_of_week="mon-fri")
    scheduler.start()

    yield

    scheduler.shutdown(wait=False)


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(router, tags=["prediction"])
app.include_router(auth_router)


@app.get("/health")
def health():
    """Health check for proxies/load balancers."""
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """Return JSON for 5xx; let FastAPI handle HTTPException (4xx) with JSON detail."""
    if isinstance(exc, StarletteHTTPException):
        raise exc
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or "Internal server error"},
    )

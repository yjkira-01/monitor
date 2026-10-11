from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..llm import config, now_ms, summarize, update_config
from ..models import Alert, BehaviorEvent, VitalRecord
from ..video import video_manager
from ..data_source import get_source

router = APIRouter(prefix="/insights", tags=["insights"])


class SummaryRequest(BaseModel):
    hours: int = Field(default=24, ge=1, le=168)
    question: str = Field(default="", max_length=500)


class LlmConfigRequest(BaseModel):
    base_url: str = Field(default="https://api.openai.com/v1", min_length=1, max_length=500)
    model: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=500)
    clear_api_key: bool = False


def _context(hours: int, question: str, db: Session) -> dict:
    cutoff = now_ms() - hours * 60 * 60 * 1000
    alert_query = db.query(Alert).filter(Alert.timestamp_ms >= cutoff)
    alerts = alert_query.order_by(Alert.timestamp_ms.desc()).limit(10).all()
    vitals = db.query(VitalRecord).filter(VitalRecord.timestamp_ms >= cutoff).count()
    actions = db.query(BehaviorEvent).filter(BehaviorEvent.timestamp_ms >= cutoff).count()
    video = video_manager.analysis()
    detection = video.get('people_detection') or {}
    video['people_detection'] = {**{key: value for key, value in detection.items() if key != 'people'},
        'people': [{key: p.get(key) for key in ('index','track_id','bbox','observation','analysis','pose_analysis')} for p in detection.get('people', [])]}
    return {
        "hours": hours,
        "question": question,
        "alerts": alert_query.count(),
        "unhandled_alerts": alert_query.filter(Alert.is_handled.is_(False)).count(),
        "recent_alerts": [
            {"level": item.level.value if hasattr(item.level, "value") else str(item.level), "type": item.type, "message": item.message}
            for item in alerts[:10]
        ],
        "vitals": vitals,
        "actions": actions,
        "vitals_source": get_source(),
        "video": {
            "action": video.get("action"),
            "fall_score": video.get("fall_score"),
            "confidence": video.get("confidence"),
            "is_fall": video.get("is_fall"),
            "watch": video.get("watch"),
            "backend": video.get("analysis_backend"),
            **{key: video.get(key) for key in ("model", "scope", "source_mode", "source_label", "model_state", "state", "assessment",
                "window_start_s", "window_end_s", "result_age_ms", "inference_ms", "feature_change", "quality", "stale", "playback_ended",
                "windows_analyzed", "consecutive_fall_windows", "short_fall_score", "short_consecutive_fall_windows", "decision_basis", "people_detection", "activity", "activity_classes", "model_revision")},
            "recent_events": video.get("events", [])[-8:],
            "recent_windows": video.get("history", [])[-8:],
            "clip_classification": {
                key: value for key, value in (video_manager.classification() or {}).items()
                if key in {"model", "scope", "predicted", "fall_probability", "activity_scores", "note"}
            },
        },
    }


@router.get("/config")
def get_config(_=Depends(get_current_user)):
    settings = config()
    return {
        "configured": settings["configured"],
        "model": settings["model"],
        "base_url": settings["base_url"],
        "api_key_set": settings["api_key_set"],
        "api_key_masked": settings["api_key_masked"],
        "runtime": settings["runtime"],
    }


@router.post("/config")
def set_config(body: LlmConfigRequest, _=Depends(get_current_user)):
    try:
        settings = update_config(
            base_url=body.base_url,
            model=body.model,
            api_key=body.api_key,
            clear_api_key=body.clear_api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "saved": True,
        "configured": settings["configured"],
        "model": settings["model"],
        "base_url": settings["base_url"],
        "api_key_set": settings["api_key_set"],
        "api_key_masked": settings["api_key_masked"],
        "runtime": settings["runtime"],
    }


@router.post("/summary")
def summary(body: SummaryRequest, db: Session = Depends(get_db), _=Depends(get_current_user)):
    context = _context(body.hours, body.question, db)
    text, provider, error = summarize(context)
    return {
        "summary": text,
        "provider": provider,
        "model": config()["model"],
        "generated_at_ms": now_ms(),
        "error": error,
        "context": context,
    }

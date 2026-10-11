from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Alert, AlertLevel
from ..schemas import AlertOut

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertHandling(BaseModel):
    is_handled: bool


@router.patch("/{alert_id}", response_model=AlertOut)
def handle_alert(alert_id: int, body: AlertHandling, db: Session = Depends(get_db), _=Depends(get_current_user)):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "预警不存在")
    alert.is_handled = body.is_handled
    db.commit()
    db.refresh(alert)
    return alert


@router.get("", response_model=list[AlertOut])
def list_alerts(limit: int = 50, db: Session = Depends(get_db), _=Depends(get_current_user)):
    return db.query(Alert).order_by(Alert.timestamp_ms.desc()).limit(limit).all()


@router.post("/trigger", response_model=AlertOut)
def trigger_alert(db: Session = Depends(get_db), _=Depends(get_current_user)):
    alert = Alert(timestamp_ms=0, level=AlertLevel.yellow, type="test", message="测试预警", is_handled=False)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

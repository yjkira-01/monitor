from pathlib import Path
from typing import Literal
import subprocess
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, Response

from ..auth import get_current_user, decode_token
from ..video import video_manager

router = APIRouter(prefix="/video", tags=["video"])
ROOT = Path(__file__).resolve().parents[3]
SAMPLES = {
    "fall": (ROOT / "data/raw/GMDCSA24/Subject 4/Fall/01.mp4", "跌倒示例 · Subject 4"),
    "adl": (ROOT / "data/raw/GMDCSA24/Subject 4/ADL/01.mp4", "正常活动 · Subject 4"),
}


class VideoSourceRequest(BaseModel):
    sample_id: Literal["fall", "adl"]


class PlaybackRequest(BaseModel):
    paused: bool


def media_auth(request: Request):
    token = request.query_params.get("token") or request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        raise HTTPException(401, "请先登录")
    decode_token(token)


@router.get("/samples")
def samples(_=Depends(get_current_user)):
    return [{"id": key, "name": name} for key, (path, name) in SAMPLES.items() if path.is_file()]


@router.post("/source")
def set_source(body: VideoSourceRequest, _=Depends(get_current_user)):
    try:
        video_manager.set_file_source(*SAMPLES[body.sample_id])
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return video_manager.status()


@router.post("/upload")
async def upload(file: UploadFile, _=Depends(get_current_user)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "请选择 MP4/AVI/MOV/MKV/WebM 视频")
    directory = ROOT / "data/uploads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}{suffix}"
    try:
        size = 0
        with path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 200 * 1024 * 1024:
                    raise HTTPException(413, "单个视频最大 200MB")
                output.write(chunk)
        video_manager.set_file_source(path, Path(file.filename).name[:120])
    except Exception as exc:
        path.unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            raise HTTPException(400, str(exc)) from exc
        if isinstance(exc, RuntimeError):
            raise HTTPException(409, str(exc)) from exc
        raise
    finally:
        await file.close()
    return video_manager.status()


@router.post("/camera")
def camera(_=Depends(get_current_user)):
    try:
        video_manager.set_camera_source()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return video_manager.status()


@router.post("/playback")
def playback(body: PlaybackRequest, _=Depends(get_current_user)):
    try:
        video_manager.set_paused(body.paused)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return video_manager.status()


@router.post("/restart")
def restart(_=Depends(get_current_user)):
    if video_manager.analysis()["source_mode"] != "file_replay":
        raise HTTPException(400, "请先选择回放视频")
    try:
        video_manager.set_file_source(Path(video_manager.source), video_manager.status()["source"])
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return video_manager.status()


@router.get("/attention")
def attention(_=Depends(media_auth)):
    jpeg = video_manager.worker.attention()
    if jpeg is None:
        raise HTTPException(404, "尚无完成窗口")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/classify")
def classify(_=Depends(get_current_user)):
    try:
        return video_manager.classify_file()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "视频分析超时，请缩短视频后再试") from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/observation")
def observation(_=Depends(media_auth)):
    jpeg = video_manager.pose_worker.observation()
    if jpeg is None:
        raise HTTPException(404, "尚无人物检测画面")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/status")
def status(_=Depends(get_current_user)):
    return video_manager.status()


@router.get("/analysis")
def analysis(_=Depends(get_current_user)):
    return video_manager.analysis()


@router.get("/pose")
def pose(_=Depends(get_current_user)):
    return video_manager.pose()


@router.get("/feed")
def feed(annotated: bool = False, _=Depends(media_auth)):
    return StreamingResponse(video_manager.stream(annotated), media_type="multipart/x-mixed-replace; boundary=frame")

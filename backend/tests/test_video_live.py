import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.auth import create_access_token, get_current_user
from app.database import Base, get_db
from app.models import Alert
from app.llm import local_summary
from app.main import app, should_persist_video_alert
from app.routers import video as routes
from app.video import VideoManager, frame_quality
from app.video_model import VideoModelWorker


def prediction(score=0.8):
    return {"fall_score": score, "embedding": np.array([1., 2.]), "attention_jpeg": b"jpeg", "attention_class": "fall", "device": "cpu"}


def accept(worker, score=0.8, generation=0, usable=True):
    end = worker.snapshot()["windows_analyzed"] + 2
    return worker._accept(prediction(score), 0, end, generation, {"usable": usable}, int(time.time()*1000), 12)


def test_window_transitions_and_quality_do_not_invent_sleep():
    worker = VideoModelWorker(Path("unused"))
    accept(worker)
    assert worker.snapshot()["state"] == "fall_candidate"
    accept(worker)
    assert worker.snapshot()["is_fall"] is True
    assert worker.snapshot()["event_id"] == 1
    accept(worker)
    assert len(worker.snapshot()["events"]) == 2
    accept(worker, usable=False)
    assert worker.snapshot()["state"] == "quality_limited"
    assert worker.snapshot()["is_fall"] is False
    accept(worker, score=0.1)
    assert worker.snapshot()["action"] == "normal_activity"
    assert worker.snapshot()["feature_change"] == 0


def test_reset_rejects_inflight_results_and_clears_evidence():
    worker = VideoModelWorker(Path("unused"))
    session_id = worker.snapshot()["session_id"]
    assert session_id and VideoModelWorker(Path("unused")).snapshot()["session_id"] != session_id
    accept(worker)
    generation = worker.reset()
    assert worker.snapshot()["session_id"] == session_id
    assert not accept(worker, generation=0)
    assert worker.snapshot()["history"] == []
    assert worker.attention() is None
    assert accept(worker, score=0.1, generation=generation)
    assert worker.snapshot()["predicted"] == "adl"


def test_duplicate_window_cannot_count_as_continuous_evidence():
    worker = VideoModelWorker(Path("unused"))
    assert accept(worker)
    assert not worker._accept(prediction(), 0, 2, 0, {}, int(time.time()*1000), 12)
    assert worker.snapshot()["consecutive_fall_windows"] == 1


def test_recent_video_requires_pose_support_and_two_continuous_windows():
    worker = VideoModelWorker(Path('unused'))
    recent = {**prediction(.1), 'short_fall_score': .8}
    quality = {'usable': True, 'short_window_eligible': True, 'pose_fall_people': [{'track_id': 1}]}
    assert worker._accept(recent, 0, 5, 0, quality, 0, 12)
    assert not worker.snapshot()['is_fall']
    assert worker._accept(recent, 0, 5.8, 0, quality, 0, 12)
    result = worker.snapshot()
    assert result['is_fall'] and result['action'] == 'falling'
    assert result['fall_score'] == .1 and result['short_fall_score'] == .8
    assert result['decision_basis'] == 'supported_short_window'
    # A long gap is a new observation, not a second confirming window.
    assert worker._accept(recent, 0, 10, 0, quality, 0, 12)
    assert not worker.snapshot()['is_fall']
    for end in (10.8, 11.6):
        assert worker._accept(recent, 0, end, 0, {'usable': True}, 0, 12)
        assert not worker.snapshot()['is_fall']
    generation = worker.reset()
    assert worker.snapshot()['short_fall_score'] is None
    assert not worker._accept(recent, 0, 20, generation-1, quality, 0, 12)


def test_recent_person_video_does_not_transfer_evidence_between_ids():
    worker = VideoModelWorker(Path('unused'))
    active = [dict(track_id=1,start_s=0,end_s=5,quality={'spatial_valid':True,'short_window_eligible':True}),
              dict(track_id=2,start_s=0,end_s=5,quality={'spatial_valid':True,'short_window_eligible':False})]
    recent = {**prediction(.1), 'short_fall_score': .9}
    worker._accept_people(active, active, [recent, recent], 0)
    for person in active:
        person['end_s'] = 5.8
    worker._accept_people(active, active, [recent, recent], 0)
    results = worker.person_predictions()
    assert results[1]['state'] == 'fall_sustained'
    assert results[2]['state'] == 'adl' and results[2]['short_consecutive_fall_windows'] == 0


def test_current_person_readout_preserves_confirmed_recent_video(monkeypatch):
    manager = VideoManager('__unused_video__.mp4')
    manager._status.update(connected=True, paused=True)
    person = {'track_id': 1, 'observation': {'usable': True, 'video_usable': True}, 'pose_analysis': {'state': 'fall_candidate'}}
    monkeypatch.setattr(manager.pose_worker, 'snapshot', lambda: {'state':'ready', 'captured_at_ms':int(time.time()*1000), 'people':[dict(person)]})
    monkeypatch.setattr(manager.worker, 'person_predictions', lambda: {1:{'fall_score':.1, 'short_fall_score':.8, 'short_consecutive_fall_windows':2, 'state':'fall_sustained', 'updated_at_ms':int(time.time()*1000)}})
    assert manager.pose()['people'][0]['analysis']['state'] == 'fall_sustained'


def test_slow_inference_keeps_only_latest_queued_window():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    class Predictor:
        def __init__(self, path):
            pass

        def predict(self, frames):
            calls.append(frames[0])
            if len(calls) == 1:
                entered.set()
                assert release.wait(3)
            else:
                completed.set()
            return prediction()

    worker = VideoModelWorker(Path("unused"), Predictor)
    worker.start()
    try:
        worker.submit([1], 0, 1, 0, {}, 1)
        assert entered.wait(2)
        worker.submit([2], 0, 2, 0, {}, 2)
        worker.submit([3], 0, 3, 0, {}, 3)
        release.set()
        assert completed.wait(2)
    finally:
        release.set()
        worker.stop()
    assert calls == [1, 3]


def test_short_replay_is_paced_and_retains_final_window(tmp_path, monkeypatch):
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    for i in range(15):
        writer.write(np.full((60, 80, 3), 60+i, dtype=np.uint8))
    writer.release()
    manager = VideoManager(str(path))
    jobs = []
    monkeypatch.setattr(manager.worker, "start", lambda: None)
    monkeypatch.setattr(manager.pose_worker, "start", lambda: None)
    manager.worker._result["model_state"] = "ready"
    monkeypatch.setattr(manager.worker, "submit", lambda *job: jobs.append(job))
    started = time.monotonic()
    manager.start()
    try:
        manager._thread.join(timeout=4)
        status = manager.status()
        assert status["ended"] and status["connected"]
        assert status["frames"] == 15
        assert time.monotonic()-started >= 1.3
        assert jobs[-1][2] == 1.4
        assert len(jobs[-1][0]) == 15
    finally:
        manager.stop()


def test_replay_and_stale_windows_cannot_persist_live_alerts():
    data = {"source_mode": "live", "model_state": "ready", "is_fall": True, "stale": False}
    assert should_persist_video_alert(data)
    assert not should_persist_video_alert({**data, "source_mode": "file_replay"})
    assert not should_persist_video_alert({**data, "stale": True})
    assert not should_persist_video_alert({**data, "model_state": "error"})


@pytest.mark.parametrize('fps',[15.,15.1])
def test_replay_displays_while_model_loads_and_15fps_timestamps_do_not_halve_pose_updates(tmp_path, monkeypatch, fps):
    from app import video
    path=tmp_path/'cold.mp4';path.touch()
    frame=np.full((60,80,3),100,np.uint8)
    class Capture:
        index=0
        def isOpened(self):return True
        def getBackendName(self):return 'fixture'
        def get(self,prop):
            return {cv2.CAP_PROP_FPS:fps,cv2.CAP_PROP_FRAME_COUNT:16,cv2.CAP_PROP_FRAME_WIDTH:80,
                    cv2.CAP_PROP_FRAME_HEIGHT:60,cv2.CAP_PROP_POS_MSEC:(self.index-1)*1000/fps}.get(prop,0)
        def read(self):
            self.index+=1
            return (True,frame) if self.index<=16 else (False,None)
        def release(self):pass
    monkeypatch.setattr(video,'open_capture',lambda source:Capture())
    manager=VideoManager(str(path));poses=[];windows=[]
    monkeypatch.setattr(manager.worker,'start',lambda:None)
    monkeypatch.setattr(manager.pose_worker,'start',lambda:None)
    monkeypatch.setattr(manager.pose_worker,'submit',lambda *args:poses.append(args[1]))
    monkeypatch.setattr(manager.worker,'submit',lambda *args:windows.append(args))
    started=time.monotonic();manager.start()
    try:
        deadline=started+.5
        while not manager._status['connected'] and time.monotonic()<deadline:time.sleep(.005)
        assert manager._status['connected'] and manager.worker.snapshot()['model_state']=='loading'
        manager._thread.join(3)
        assert manager._status['ended'] and len(set(round(p,5) for p in poses))>=15
        assert windows[-1][2]==pytest.approx(15/fps), 'the loading model still receives the retained final window'
    finally:manager.stop()


def test_quality_and_summary_keep_provenance():
    quality, _ = frame_quality(np.zeros((60, 80, 3), dtype=np.uint8))
    assert quality["usable"] is False
    context = {"hours": 24, "alerts": 0, "unhandled_alerts": 0, "vitals": 1, "actions": 0, "vitals_source": "mock",
               "video": {"source_mode": "file_replay", "is_fall": True, "window_start_s": 0., "window_end_s": 4.6, "windows_analyzed": 5}}
    result = local_summary(context)
    assert "回放测试" in result and "mock" in result
    assert "4.6" in result and "现场告警" in result


@pytest.mark.parametrize('label', ['adl','fall','lying','stand_up','invented'])
def test_whole_video_classification_accepts_trained_states_and_rejects_unknown(tmp_path, monkeypatch, label):
    import json
    from types import SimpleNamespace
    from app import video
    path = tmp_path/'test.mp4';path.touch()
    manager = VideoManager(str(path))
    weight = tmp_path/'inference.pt';weight.touch();manager.worker.checkpoint = weight
    result = {'scope':'whole_video','predicted':label,'fall_probability':.1,'video':'private-local-path'}
    monkeypatch.setattr(video.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=json.dumps(result)))
    if label == 'invented':
        with pytest.raises(RuntimeError,match='无效分类'):manager.classify_file()
    else:
        assert manager.classify_file() == {k:v for k,v in result.items() if k!='video'}
    assert not manager._classification_lock.locked()


def test_video_api_auth_source_boundary_and_invalid_upload(tmp_path, monkeypatch):
    client = TestClient(app)
    assert client.get("/api/video/status").status_code == 401
    assert client.get("/api/video/pose").status_code == 401
    assert client.post("/api/video/camera").status_code == 401
    assert client.get("/api/video/feed").status_code == 401
    assert client.get("/api/video/attention?token=bad").status_code == 401
    assert client.get("/api/video/observation?token=bad").status_code == 401
    assert client.get("/api/edge/point-cloud").status_code == 401
    app.dependency_overrides[get_current_user] = lambda: object()
    monkeypatch.setattr(routes, "ROOT", tmp_path)
    try:
        assert client.post("/api/video/source", json={"sample_id": "../../secret"}).status_code == 422
        assert client.post("/api/video/upload", files={"file": ("test.exe", b"invalid")}).status_code == 400
        response = client.post("/api/video/upload", files={"file": ("bad.mp4", b"invalid")})
        assert response.status_code == 400
        assert list((tmp_path / "data/uploads").iterdir()) == []
        token = create_access_token("test", "admin")
        monkeypatch.setattr(routes.video_manager.worker, "attention", lambda: b"jpeg")
        response = client.get(f"/api/video/attention?token={token}")
        assert response.content == b"jpeg" and response.headers["cache-control"] == "no-store"
        monkeypatch.setattr(routes.video_manager.pose_worker, "observation", lambda: b"people-jpeg")
        response = client.get(f"/api/video/observation?token={token}")
        assert response.content == b"people-jpeg" and response.headers["cache-control"] == "no-store"
    finally:
        app.dependency_overrides.clear()


def test_camera_api_returns_live_status_and_explains_unavailable_device(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: object()
    calls = []
    monkeypatch.setattr(routes.video_manager, 'set_camera_source', lambda: calls.append('camera'))
    monkeypatch.setattr(routes.video_manager, 'status', lambda: {'source_mode':'live','source':'EMEET C60E · USB'})
    try:
        client = TestClient(app)
        response = client.post('/api/video/camera')
        assert response.status_code == 200 and response.json()['source_mode'] == 'live' and calls == ['camera']
        def unavailable(): raise ValueError('USB 摄像头不可用')
        monkeypatch.setattr(routes.video_manager, 'set_camera_source', unavailable)
        assert client.post('/api/video/camera').status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_handled_alert_is_saved_and_missing_ids_return_404():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = Session(engine)
    alert = Alert(timestamp_ms=1, level="yellow", type="test", message="test", is_handled=False)
    db.add(alert)
    db.commit()
    identifier = alert.id
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: object()
    try:
        client = TestClient(app)
        response = client.patch(f"/api/alerts/{identifier}", json={"is_handled": True})
        assert response.status_code == 200
        db.expire_all()
        assert db.get(Alert, identifier).is_handled
        assert client.patch("/api/alerts/99999", json={"is_handled": True}).status_code == 404
        from app.routers.insights import _context
        db.add_all([Alert(timestamp_ms=int(time.time()*1000), level="yellow", type="test", message="sample", is_handled=False) for _ in range(105)])
        db.commit()
        context = _context(24, "", db)
        assert context["alerts"] == 105 and context["unhandled_alerts"] == 105
        assert len(context["recent_alerts"]) == 10
    finally:
        app.dependency_overrides.clear()
        db.close()
        engine.dispose()


def test_live_websocket_requires_valid_login_token():
    client = TestClient(app)
    for path in ("/ws", "/ws/stream"):
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(path + "?token=invalid"):
                pass
        assert rejected.value.code == 1008
    token = create_access_token("admin", "admin")
    with client.websocket_connect(f"/ws/stream?token={token}") as socket:
        socket.send_text("heartbeat")

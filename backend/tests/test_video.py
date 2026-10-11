import cv2
import pytest
from app import video
from app.video import VideoManager, open_capture


def test_preview_bounds_bandwidth_without_modifying_source_or_changing_aspect():
    import numpy as np
    frame=np.full((1080,1920,3),137,np.uint8)
    preview=video.preview_frame(frame)
    assert preview.shape==(540,960,3) and frame.shape==(1080,1920,3)
    assert np.all(frame==137) and np.all(preview==137)
    native=np.zeros((480,640,3),np.uint8)
    assert video.preview_frame(native) is native


def test_paused_annotated_stream_shows_detection_that_completed_after_capture(monkeypatch):
    manager=VideoManager('__missing__.mp4')
    monkeypatch.setattr(manager,'start',lambda:None)
    manager._status.update(paused=True,ended=True)
    monkeypatch.setattr(manager.pose_worker,'observation',lambda:b'latest-boxes')
    assert b'latest-boxes' in next(manager.stream(annotated=True))
    assert b'latest-boxes' not in next(manager.stream(annotated=False))


def test_video_source_failure_still_serves_placeholder_frame(monkeypatch):
    manager = VideoManager("__missing_video_source__.mp4")
    monkeypatch.setattr(manager.worker, "start", lambda: None)
    monkeypatch.setattr(manager.pose_worker, "start", lambda: None)
    try:
        status = manager.status()
        assert status["connected"] is False
        chunk = next(manager.stream())
        assert chunk.startswith(b"--frame")
        assert b"Content-Type: image/jpeg" in chunk
    finally:
        manager.stop()


def test_usb_capture_negotiates_low_resolution_but_does_not_modify_files(monkeypatch):
    calls, settings = [], []
    class Capture:
        def isOpened(self): return True
        def get(self, prop): return 0
        def set(self, prop, value): settings.append((prop, value)); return True
    def factory(*args):
        calls.append(args)
        return Capture()
    monkeypatch.setattr(cv2, 'VideoCapture', factory)
    open_capture(1)
    assert calls[-1] == (1, cv2.CAP_DSHOW if video.os.name == 'nt' else cv2.CAP_ANY)
    assert (cv2.CAP_PROP_FRAME_WIDTH,640) in settings and (cv2.CAP_PROP_FRAME_HEIGHT,480) in settings and (cv2.CAP_PROP_FPS,30) in settings
    if video.os.name == 'nt':
        assert (cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')) in settings
        assert (cv2.CAP_PROP_EXPOSURE,-5) in settings and (cv2.CAP_PROP_GAIN,64) in settings
        assert (cv2.CAP_PROP_ZOOM,100) in settings
    settings.clear()
    open_capture('recording.mp4')
    assert calls[-1] == ('recording.mp4',) and settings == []


def test_unsupported_camera_controls_restore_auto_exposure(monkeypatch):
    if video.os.name != 'nt': pytest.skip('C60E DirectShow controls are Windows only')
    settings = []
    class Capture:
        def isOpened(self): return True
        def get(self, prop): return 12
        def set(self, prop, value): settings.append((prop,value)); return prop != cv2.CAP_PROP_EXPOSURE
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: Capture())
    open_capture(1)
    assert settings[-2:] == [(cv2.CAP_PROP_GAIN,12),(cv2.CAP_PROP_AUTO_EXPOSURE,1)]


def test_camera_source_resets_replay_evidence_and_failed_open_keeps_old_source(monkeypatch):
    manager = VideoManager('old.mp4')
    monkeypatch.setattr(manager, 'start', lambda: None)
    class Capture:
        opened = True
        def isOpened(self): return self.opened
        def read(self): return True, object()
        def release(self): pass
    cap = Capture()
    monkeypatch.setattr(video, 'open_capture', lambda source: cap)
    cap.opened = False
    with pytest.raises(ValueError, match='无法读取 C60E'):
        manager.set_camera_source()
    assert manager.source == 'old.mp4' and manager._generation == 0
    cap.opened = True
    manager._status.update(paused=True,ended=True,frames=15,position_s=2,duration_s=4,width=1920,height=1080)
    manager.set_camera_source()
    assert manager.source == manager.camera_index
    assert manager._generation == 1 and manager.pose_worker.snapshot()['generation'] == 1
    assert manager.pose()['session_id'] == manager.analysis()['session_id'] == manager.worker.session_id
    assert manager._status['source_mode'] == 'live'
    assert not manager._status['paused'] and not manager._status['ended']
    assert manager._status['frames'] == 0 and manager._status['position_s'] == 0 and manager._status['duration_s'] == 0
    assert manager._status['width'] == 0 and manager._status['height'] == 0
    manager.set_camera_source()
    assert manager._generation == 2, 'reconnect must reopen, without preflight opening an occupied camera'

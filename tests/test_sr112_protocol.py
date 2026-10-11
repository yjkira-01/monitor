"""Independent wire layouts from the supplied TI GUI; all packets here are synthetic."""
import asyncio
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from app.sources.oob_tlv import parse_frame, consume_frames, build_oob_frame
from app.sources import mmvital_source
from app.sources.mmvital_source import MmVitalSource, RadarPhaseProvider


def packet(tlvs, n_points=0, frame_number=17):
    body = b''.join(struct.pack('<II', kind, len(payload))+payload for kind, payload in tlvs)
    length = (40+len(body)+31)//32*32
    header = struct.pack('<8s8I', bytes.fromhex('0201040306050807'), 0x03060000, length, 0xA6843,
                         frame_number, 100000, n_points, len(tlvs), 0)
    return (header+body).ljust(length, b'\0')


def track(target_id=0, velocity=0):
    return struct.pack('<I27f', target_id, 0., 1., 1., velocity, 0., 0., *([0.]*19), 1., .9)


def vital(deviation=.1, target_id=0):
    return struct.pack('<HH33f', target_id, 21, deviation, 66., 15., *range(15), *range(100, 115))


def hardware_packet(deviation=.1, target_id=0, tracks=True, velocity=0):
    tlvs = [(1010, track(0, velocity))] if tracks else []
    return packet(tlvs+[(1040, vital(deviation, target_id))])


def provider(monkeypatch, chunks):
    chunks = list(chunks)
    reader = SimpleNamespace(read=lambda _: chunks.pop(0) if chunks else b'', open=lambda: None)
    monkeypatch.setenv('RADAR_PORT', 'test-only-no-device')
    monkeypatch.setattr(mmvital_source, 'serial', SimpleNamespace(Serial=lambda *args, **kwargs: reader))
    return RadarPhaseProvider()


def test_data_uart_disables_modem_lines_before_open(monkeypatch):
    calls = []
    class Reader:
        def open(self):
            calls.append((self.port, self.dtr, self.rts))
    def create(**kwargs):
        assert kwargs == {'port': None, 'baudrate': 921600, 'timeout': .2}
        return Reader()
    monkeypatch.setenv('RADAR_PORT', 'test-only-no-device')
    monkeypatch.setattr(mmvital_source, 'serial', SimpleNamespace(Serial=create))
    p = RadarPhaseProvider()
    assert calls == [('test-only-no-device', False, False)]
    assert p._enabled and p.error is None


def test_sparse_uart_reads_only_available_bytes(monkeypatch):
    p = provider(monkeypatch, [])
    wire = hardware_packet()
    def read(size):
        assert size == p._serial.in_waiting
        return wire if size else b''
    p._serial.read = read
    p._serial.in_waiting = 0
    assert p._poll_frames() == []
    p._serial.in_waiting = len(wire)
    assert len(p._poll_frames()) == 1


def test_sdk3_points_have_no_count_prefix_and_builder_uses_standard_wire_layout():
    data = struct.pack('<8f', .1, .2, .3, -.4, 1., 2., 3., .5)
    parsed = parse_frame(packet([(1, data)], n_points=2))
    assert parsed is not None and parsed.frame_number == 17 and parsed.tlv_types == (1,)
    np.testing.assert_allclose(parsed.points, [[.1, .2, .3], [1, 2, 3]])
    assert parse_frame(packet([(1, struct.pack('<I', 2)+data)], n_points=2)) is None
    built = build_oob_frame(np.array([[1., 2., 3.]]))
    assert len(built) % 32 == 0 and struct.unpack_from('<II', built, 40) == (1, 16)
    assert struct.unpack_from('<4f', built, 48) == (1., 2., 3., 0.)


def test_compressed_spherical_points_use_signed_angles_and_radar_axes():
    units = struct.pack('<5f', np.pi/2, np.pi/2, .1, .01, .1)
    points = b''.join(struct.pack('<bbhHH', *p) for p in [(0, 0, 0, 100, 10), (0, -1, -4, 200, 20), (1, 0, 0, 100, 10)])
    frame = parse_frame(packet([(1020, units+points)], n_points=3))
    np.testing.assert_allclose(frame.points, [[0, 1, 0], [-2, 0, 0], [0, 0, 1]], atol=1e-6)


def test_vital_packet_extracts_136_bytes_and_does_not_invent_millimeters():
    frame = parse_frame(hardware_packet())
    assert frame.tracks[0]['target_id'] == 0 and frame.tracks[0]['position'] == [0., 1., 1.]
    v = frame.vitals[0]
    assert v['heart_rate'] == 66 and v['breath_rate'] == 15 and v['range_bin'] == 21
    assert v['heart_waveform'] == list(range(15)) and v['breath_waveform'] == list(range(100, 115))
    assert 'chest_displacement_mm' not in v


def test_all_fragment_boundaries_keep_partial_frames_and_magic():
    first = packet([(1, struct.pack('<4f', 1., 2., 3., 0.))], n_points=1)
    second = hardware_packet()
    for cut in range(1, len(first+second)):
        frames, remainder = consume_frames(b'junk'+(first+second)[:cut])
        rest, tail = consume_frames(remainder+(first+second)[cut:])
        assert len(frames+rest) == 2 and len(tail) <= 7, cut
        np.testing.assert_allclose((frames+rest)[0].points, [[1, 2, 3]])
        assert (frames+rest)[1].vitals[0]['heart_rate'] == 66


@pytest.mark.parametrize('kind,payload,count', [(1, b'x', 1), (1, struct.pack('<4f', float('nan'), 0, 0, 0), 1),
                                                (1020, b'x'*19, 1), (1010, b'x'*111, 0), (1040, b'x'*135, 0),
                                                (1040, vital(float('inf')), 0)])
def test_bad_known_tlv_is_rejected_and_stream_recovers(kind, payload, count):
    bad = packet([(kind, payload)], n_points=count)
    assert parse_frame(bad) is None
    frames, _ = consume_frames(bad+hardware_packet())
    assert len(frames) == 1 and frames[0].vitals[0]['heart_rate'] == 66


def test_corrupt_lengths_reject_without_reading_past_frame():
    good = hardware_packet()
    for offset, value in [(12, 10), (12, 2**32-1), (32, 2**32-1), (44, 2**32-1)]:
        bad = bytearray(good)
        struct.pack_into('<I', bad, offset, value)
        assert parse_frame(bytes(bad)) is None
        frames, _ = consume_frames(bytes(bad)+good)
        assert len(frames) == 1


def test_no_device_never_falls_back_to_synthetic_real_data(monkeypatch):
    monkeypatch.delenv('RADAR_PORT', raising=False)
    source = MmVitalSource()
    assert isinstance(source.provider, RadarPhaseProvider) and source.next_vital() is None


def test_cloud_only_updates_cloud_but_cannot_estimate_heart_rate(monkeypatch):
    wire = packet([(1, struct.pack('<4f', 1., 2., 3., 0.))], n_points=1)
    p = provider(monkeypatch, [wire[:51], wire[51:]])
    assert p.next_vital() is None and p._latest_points is None
    assert p.next_vital() is None
    np.testing.assert_allclose(p._latest_points, [[1, 2, 3]])
    assert p.next_window() is None


def test_firmware_vitals_keep_provenance_and_cannot_repeat_stale_results(monkeypatch):
    p = provider(monkeypatch, [hardware_packet(velocity=.2)])
    result = MmVitalSource(device_id='RADAR_TEST', provider=p).next_vital()
    assert result['source'] == 'real' and result['signal_type'] == 'ti_vitals_tlv1040'
    assert result['device_id'] == 'RADAR_TEST' and result['target_id'] == 0
    assert result['motion_flag'] and result['needs_review'] and result['quality'] is None
    assert result['chest_displacement_mm'] is None and result['ahi_index'] is None and result['vital_agreement'] is None
    assert p.next_vital() is None


@pytest.mark.parametrize('kwargs', [{'tracks':False}, {'target_id':1}, {'deviation':0}])
def test_missing_patient_cannot_emit_vitals(monkeypatch, kwargs):
    assert provider(monkeypatch, [hardware_packet(**kwargs)]).next_vital() is None


def test_weak_breath_is_missing_rate_not_apnea_diagnosis(monkeypatch):
    result = provider(monkeypatch, [hardware_packet(deviation=.01)]).next_vital()
    assert result['breath_rate'] is None and result['heart_rate'] == 66 and result['ahi_index'] is None
    assert result['needs_review'] and '不是呼吸暂停诊断' in result['watch_reason']


def test_absence_and_expired_packet_clear_pending_vitals(monkeypatch):
    p = provider(monkeypatch, [hardware_packet()+hardware_packet(tracks=False)])
    assert p.next_vital() is None
    p = provider(monkeypatch, [hardware_packet()])
    p._poll_frames()
    v, t, captured, frame = p._pending_vitals.popleft()
    p._pending_vitals.append((v, t, captured-4000, frame))
    assert p.next_vital() is None


def test_radar_snapshot_is_read_only_coherent_and_retains_waveforms_between_updates(monkeypatch):
    from app import data_source
    wire = packet([(1, struct.pack('<4f', .5, 2., 1., 0.)), (1010, track()), (1040, vital())], n_points=1)
    p = provider(monkeypatch, [wire])
    monkeypatch.setattr(data_source._real, 'provider', p)
    monkeypatch.setattr(data_source, 'get_source', lambda: 'real')
    p._poll_frames()
    captured = p._telemetry['timestamp_ms']
    monkeypatch.setattr(data_source.time, 'time', lambda: (captured+200)/1000)
    monkeypatch.setattr(p._serial, 'read', lambda _: pytest.fail('snapshot must not read serial'))
    snapshot = data_source.radar_snapshot()
    assert snapshot['status'] == 'tracking' and snapshot['age_ms'] == 200
    assert snapshot['points'] == [[.5, 2., 1.]] and snapshot['tracks'][0]['target_id'] == 0
    assert snapshot['vitals'][0]['heart_waveform'] == list(range(15))
    assert len(p._pending_vitals) == 1 and snapshot['quality'] is None and not snapshot['calibrated']
    monkeypatch.setattr(p._serial, 'read', lambda _: packet([(1010, track(velocity=.2))], frame_number=18))
    p._poll_frames()
    assert data_source.radar_snapshot()['frame_number'] == 18
    assert data_source.radar_snapshot()['vitals'][0]['firmware_frame_number'] == 17
    monkeypatch.setattr(data_source.time, 'time', lambda: (captured+3100)/1000)
    snapshot = data_source.radar_snapshot()
    assert snapshot['status'] == 'tracking' and snapshot['vitals'][0]['stale']
    monkeypatch.setattr(data_source.time, 'time', lambda: (captured+4000)/1000)
    assert data_source.radar_snapshot()['status'] == 'stale'


def test_radar_snapshot_target_absence_invalid_waveform_and_source_switch_clear_cache(monkeypatch):
    from app import data_source
    p = provider(monkeypatch, [hardware_packet(), hardware_packet(deviation=0), hardware_packet(), packet([])])
    monkeypatch.setattr(data_source._real, 'provider', p)
    monkeypatch.setattr(data_source, '_current_source', {'value': 'real'})
    p._poll_frames()
    assert len(data_source.radar_snapshot()['vitals']) == 1
    p._poll_frames()
    assert data_source.radar_snapshot()['vitals'] == []
    p._poll_frames()
    assert data_source.radar_snapshot()['vitals']
    p._poll_frames()
    assert data_source.radar_snapshot()['status'] == 'no_target'
    assert data_source.radar_snapshot()['tracks'] == data_source.radar_snapshot()['vitals'] == []
    old_session = p.session_id
    data_source.set_source('mock')
    snapshot = data_source.radar_snapshot()
    assert snapshot['status'] == 'inactive' and snapshot['points'] == snapshot['vitals'] == []
    assert snapshot['session_id'] != old_session and not p._pending_vitals and p._latest_points is None
    data_source.set_source('real')
    assert data_source.radar_snapshot()['status'] == 'waiting'


def test_radar_snapshot_auth_and_serial_failure(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.auth import get_current_user
    from app.routers.edge import router
    from app import data_source
    p = provider(monkeypatch, [hardware_packet()])
    monkeypatch.setattr(data_source._real, 'provider', p)
    monkeypatch.setattr(data_source, 'get_source', lambda: 'real')
    p._poll_frames()
    def fail(_):
        raise OSError('private device detail')
    monkeypatch.setattr(p._serial, 'read', fail)
    assert p.next_vital() is None
    snapshot = data_source.radar_snapshot()
    assert snapshot['status'] == 'disconnected' and snapshot['vitals'] == snapshot['points'] == []
    assert 'private' not in snapshot['hardware']['error']
    app = FastAPI()
    app.include_router(router, prefix='/api')
    client = TestClient(app)
    assert client.get('/api/edge/radar').status_code == 401
    app.dependency_overrides[get_current_user] = lambda: object()
    assert client.get('/api/edge/radar').json()['status'] == 'disconnected'


@pytest.mark.parametrize('include_vital', [False, True])
def test_real_stream_persists_columns_keeps_metadata_and_cloud_heartbeat(monkeypatch, include_vital):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app import main
    from app.database import Base
    from app.models import VitalRecord

    engine = create_engine('sqlite://', connect_args={'check_same_thread':False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    result = provider(monkeypatch, [hardware_packet()]).next_vital()
    received, touches = [], []
    async def broadcast(message, **kwargs): received.append(message)
    async def checks(): pass
    async def stop(_): raise asyncio.CancelledError
    monkeypatch.setattr(main, 'SessionLocal', sessions)
    monkeypatch.setattr(main, 'get_source', lambda: 'real')
    monkeypatch.setattr(main, 'next_real_vital', lambda: result if include_vital else None)
    monkeypatch.setattr(main, 'point_cloud_snapshot', lambda: {'stale':False, 'timestamp_ms':123})
    monkeypatch.setattr(main, 'monitor', SimpleNamespace(touch=lambda *args: touches.append(args), publish_checks=checks))
    monkeypatch.setattr(main, 'broadcast', broadcast)
    monkeypatch.setattr(main.asyncio, 'sleep', stop)
    try:
        with pytest.raises(asyncio.CancelledError): asyncio.run(main.mock_stream())
        with sessions() as db:
            assert db.query(VitalRecord).count() == int(include_vital)
        assert ('real',123) in touches
        if include_vital:
            assert received[0]['data']['heart_waveform'] == list(range(15))
            assert received[0]['data']['signal_type'] == 'ti_vitals_tlv1040'
    finally:
        engine.dispose()

"""Export an immutable InternVideo2 training checkpoint for the C# preview profile."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from yanglao_monitor.foundation_video import build_internvideo2, normalize_video, PREPROCESSING


class PreviewExport(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.net = build_internvideo2(len(state['classes']), checkpointing=False)
        self.net.load_state_dict(state['model'], strict=True)

    def forward(self, frames):
        net = self.net
        clip = normalize_video(frames.permute(0, 1, 4, 2, 3)/255.)
        x = net.patch_embed(clip).flatten(1, 2)
        x = torch.cat((net.cls_token.expand(x.shape[0], -1, -1), x), dim=1)+net.pos_embed
        for block in net.blocks:
            x = block(x)
        projector = net.clip_projector
        cross = projector.cross_attn
        q = F.linear(projector.norm1_q(x.mean(1, keepdim=True)), cross.q.weight, cross.q_bias)
        k = F.linear(projector.norm1_k(x), cross.k.weight, cross.k_bias)
        q = q.reshape(1, 1, cross.num_heads, -1).transpose(1, 2)
        k = k.reshape(1, 2049, cross.num_heads, -1).transpose(1, 2)
        attention = ((q*cross.scale)@k.transpose(-2, -1)).softmax(-1).mean(1)[:, 0, 1:]
        attention = attention.reshape(1, 8, 16, 16).mean(1)
        low, high = attention.flatten(1).amin(1)[:, None, None], attention.flatten(1).amax(1)[:, None, None]
        attention = (attention-low)/(high-low).clamp_min(1e-9)
        embedding = net.fc_norm(projector(x))
        return net.head(embedding).softmax(1), embedding, attention


def runtime_input(frames, low_light):
    selected = frames[np.linspace(0, len(frames)-1, 8).round().astype(int)]
    gamma = 1.
    if low_light:
        means = sorted(float(cv2.cvtColor(cv2.resize(f, (160, 90), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY).mean()) for f in selected)
        mean = (means[3]+means[4])*.5
        if 1 <= mean < 65:
            gamma = np.clip(np.log(80/255)/np.log(mean/255), .3, 1.)
    rgb = selected[..., [2, 1, 0]].copy()
    if gamma < 1:
        rgb = (255*np.power(rgb.astype(np.float64)/255, gamma)).astype(np.uint8)
    return rgb.astype(np.float32)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(20261009)
    checkpoint, output = Path(args.checkpoint), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['model_name'] == 'internvideo2_base' and state['preprocessing'] == PREPROCESSING
    model = PreviewExport(state).eval()
    sample = torch.rand(1, 8, 224, 224, 3)*255
    with torch.inference_mode():
        expected = model(sample)
        original = model.net(normalize_video(sample.permute(0, 1, 4, 2, 3)/255.)).softmax(1)
        torch.testing.assert_close(expected[0], original, rtol=1e-5, atol=1e-6)
    path = output/'video.onnx'
    torch.onnx.export(model, sample, str(path), input_names=['input'],
                      output_names=['probabilities', 'embedding', 'attention'], opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 3, 1
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'], sess_options=options)
    actual = session.run(None, {'input': sample.numpy()})
    errors = []
    for value, reference in zip(actual, expected):
        reference = reference.numpy()
        errors.append(float(np.max(np.abs(value-reference))))
        np.testing.assert_allclose(value, reference, atol=2e-4, rtol=2e-4)
    info = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
            'input': list(sample.shape), 'outputs': ['probabilities', 'embedding', 'attention'],
            'max_errors': errors, 'classes': state['classes'], 'read_size': [224, 224], 'clip_len': 8,
            'embedding_size': 768, 'preprocessing': PREPROCESSING, 'model_name': 'InternVideo2 Base',
            'epoch': state['epoch'], 'revision': f"stage_f_epoch_{state['epoch']}",
            'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'low_light_trained': state['low_light_trained'], 'partial_body_trained': state['partial_body_trained'],
            'acceptance': 'user_requested_training_preview', 'metrics': state['metrics'],
            'clinical_validated': False,
            'attention_semantics': 'Normalized temporal/head mean of learned pooling attention; not class activation or person location.'}
    base = json.loads((ROOT/'desktop/CareMonitor/Assets/models.json').read_text(encoding='utf-8'))
    base['video'] = info
    detector = ROOT/'artifacts/pretrained/people_rf_detr_medium.onnx'
    assert hashlib.sha256(detector.read_bytes()).hexdigest() == '8e33f2d32592918b2274a12209a80adcdebaab6b21fbc7f9eca168390dc6a72b'
    shutil.copyfile(detector, output/'people.onnx')
    base['people'] = {'sha256': hashlib.sha256(detector.read_bytes()).hexdigest(),
                      'bytes': detector.stat().st_size, 'input': [1, 3, 576, 576],
                      'box_scale': 320, 'model_name': 'RF-DETR Medium', 'person_label': 1,
                      'source': 'https://huggingface.co/Roboflow/rf-detr-medium', 'license': 'Apache-2.0'}
    (output/'models.json').write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding='utf-8')
    fixture = json.loads((ROOT/'desktop/CareMonitor/Assets/selfcheck.json').read_text())
    frames = np.array([((np.arange(224*224*3, dtype=np.int64)+t*7)%251).astype(np.uint8).reshape(224,224,3) for t in range(16)])
    fixture['video_scores'] = session.run(['probabilities'], {'input': runtime_input(frames, state['low_light_trained'])})[0].flatten().tolist()
    dark = np.array([(frame*(.08+.02*t)).astype(np.uint8) for t, frame in enumerate(frames)])
    fixture['night_scores'] = session.run(['probabilities'], {'input': runtime_input(dark, state['low_light_trained'])})[0].flatten().tolist()
    (output/'selfcheck.json').write_text(json.dumps(fixture), encoding='utf-8')
    print(json.dumps({'epoch': state['epoch'], 'max_errors': errors, 'output': str(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()

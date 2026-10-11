import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from scripts.train_video import accumulation_weight, bounded_sampling_weights
from yanglao_monitor.foundation_video import normalize_video
from yanglao_monitor._vendor.internvideo2.internvideo2 import Attention, RMSNorm, Block
from yanglao_monitor.video import _read_frames


def test_sdpa_preserves_authors_qk_normalization_and_gradients():
    torch.manual_seed(4)
    attention = Attention(48, 3, qkv_bias=True, qk_normalization=True, norm_layer=RMSNorm).eval()
    x = torch.randn(2, 17, 48, requires_grad=True)
    q, k, v = attention.qkv(x).reshape(2,17,3,3,16).permute(2,0,3,1,4).unbind(0)
    q = attention.q_norm(q.transpose(1,2).flatten(-2)).view(2,17,3,16).transpose(1,2)
    k = attention.k_norm(k.transpose(1,2).flatten(-2)).view(2,17,3,16).transpose(1,2)
    expected = attention.proj((((q*attention.scale)@k.transpose(-2,-1)).softmax(-1)@v).transpose(1,2).reshape(2,17,48))
    actual = attention(x)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=1e-5)


def test_checkpoint_backpropagates_when_input_is_frozen():
    block = Block(48, 3, norm_layer=RMSNorm, with_cp=True, init_values=1e-5, qk_normalization=True)
    result = block(torch.randn(1,17,48))
    result.square().sum().backward()
    assert torch.isfinite(block.attn.qkv.weight.grad).all()
    assert block.attn.qkv.weight.grad.abs().max() > 0


def test_bf16_sdpa_keeps_query_key_and_value_dtypes_compatible():
    attention = Attention(48,3,qk_normalization=True,norm_layer=RMSNorm)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        result = attention(torch.randn(2,17,48))
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    result.float().square().sum().backward()
    assert torch.isfinite(attention.qkv.weight.grad).all()


def test_accumulation_matches_full_batch_including_partial_tail():
    torch.manual_seed(2)
    left, right = nn.Linear(3,2), nn.Linear(3,2)
    right.load_state_dict(left.state_dict())
    inputs, targets = torch.randn(9,3), torch.randint(2,(9,))
    opt_left, opt_right = torch.optim.SGD(left.parameters(),lr=.05), torch.optim.SGD(right.parameters(),lr=.05)
    for start in (0,8):
        opt_left.zero_grad()
        nn.functional.cross_entropy(left(inputs[start:start+8]),targets[start:start+8]).backward()
        opt_left.step()
    opt_right.zero_grad()
    for index,start in enumerate(range(0,9,2)):
        x,y = inputs[start:start+2],targets[start:start+2]
        loss = nn.functional.cross_entropy(right(x),y)*accumulation_weight(index,9,len(y),2,4)
        loss.backward()
        if (index+1)%4 == 0 or index == 4:
            opt_right.step(); opt_right.zero_grad()
    for expected, actual in zip(left.parameters(),right.parameters()):
        torch.testing.assert_close(actual,expected)


def test_sampler_caps_small_domain_repetition_globally():
    rows = [{'dataset':'large','label':'walk','split':'train'} for _ in range(1000)]
    rows += [{'dataset':'rare','label':'lying','split':'train'}]
    weights = bounded_sampling_weights(rows)
    assert sum(weights) == pytest.approx(1.)
    assert max(weights)*len(rows) <= 5
    assert weights[-1] > weights[0]
    with pytest.raises(ValueError):
        bounded_sampling_weights([{'split':'test'}])


def test_letterbox_keeps_full_view_and_matches_normalization(tmp_path):
    path = tmp_path/'wide.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(80,40))
    assert writer.isOpened()
    writer.write(np.full((40,80,3),[0,0,240],dtype=np.uint8)); writer.release()
    clip = _read_frames(path,8,(224,224),preserve_aspect=True)
    assert clip.shape == (3,8,224,224)
    assert (clip[0,:,56:168,:] > .9).all()
    torch.testing.assert_close(clip[:,:,0,0], torch.tensor([124,116,104]).float().view(3,1).expand(3,8)/255)
    normalized = normalize_video(clip.transpose(0,1).unsqueeze(0))
    assert normalized.shape == (1,3,8,224,224) and torch.isfinite(normalized).all()
    with pytest.raises(ValueError):
        normalize_video(torch.zeros(1,16,3,224,224))

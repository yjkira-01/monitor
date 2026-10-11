"""OpenAI-compatible 大模型摘要接口的最小实现。

没有配置密钥时返回本地规则摘要，页面仍然可用；配置后只需替换 LLM_BASE_URL/LLM_MODEL。
"""

from __future__ import annotations

import json
import os
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse


_runtime: dict[str, str] = {}
_runtime_lock = threading.Lock()


def _masked(key: str | None) -> str | None:
    if not key:
        return None
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:4]}{'•' * 8}{key[-4:]}"


def config() -> dict[str, object]:
    with _runtime_lock:
        runtime = dict(_runtime)
    key = runtime.get("api_key") if "api_key" in runtime else (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    base_url = runtime.get("base_url") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = runtime.get("model") or os.getenv("LLM_MODEL", "gpt-4o-mini")
    return {
        "configured": bool(key),
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key_set": bool(key),
        "api_key_masked": _masked(key),
        "runtime": bool(runtime),
    }


def update_config(*, base_url: str, model: str, api_key: str | None = None, clear_api_key: bool = False) -> dict[str, object]:
    """更新当前后端进程配置；密钥只保存在内存，不写入数据库或日志。"""
    normalized_url = base_url.strip().rstrip("/")
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url 必须是 http 或 https 地址")
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model 不能为空")
    if len(normalized_model) > 200 or len(normalized_url) > 500:
        raise ValueError("配置字段过长")
    with _runtime_lock:
        _runtime["base_url"] = normalized_url
        _runtime["model"] = normalized_model
        if clear_api_key:
            _runtime["api_key"] = ""
        elif api_key is not None and api_key.strip():
            _runtime["api_key"] = api_key.strip()
    return config()


def local_summary(context: dict) -> str:
    alerts = context["alerts"]
    unhandled = context["unhandled_alerts"]
    video = context["video"]
    vitals = context["vitals"]
    actions = context["actions"]
    if video.get("model_state") == "error":
        focus = "视频训练模型不可用，不能据此判断当前活动。"
    elif video.get("stale"):
        focus = "视频结果已过期或视频源未连接，请先检查画面。"
    elif video.get('state') == 'observation_limited':
        focus = video.get('assessment', '请结合逐人观察与原视频复核。')
    elif video.get('state') == 'model_disagreement':
        focus = '姿态时序模型出现跌倒特征，RGB模型未识别，两者分歧，需要人工复核；不能据RGB低分判断安全。'
    elif video.get("is_fall"):
        focus = "视频训练模型连续窗口呈现跌倒特征，建议人工复核。"
    elif video.get("watch"):
        focus = "视频训练模型当前窗口呈现疑似跌倒特征，建议持续观察。"
    else:
        focus = "视频端侧当前未形成连续跌倒判定。"
    if video.get("source_mode") == "file_replay":
        focus = "当前为视频回放测试，结果不会生成现场告警。" + focus
    if video.get("window_end_s") is not None:
        focus += f"最近完成窗口 {video.get('window_start_s', 0):.1f}–{video['window_end_s']:.1f} 秒，共分析 {video.get('windows_analyzed', 0)} 窗口。"
    quality = video.get("quality") or {}
    if quality:
        focus += f"辅助画面观察：{quality.get('activity', '未知')}，{quality.get('note', '未知')}。"
    detection = video.get("people_detection") or {}
    if detection.get("state") == "ready":
        count = len(detection.get("people", []))
        focus += f"检测帧有 {count} 个姿态目标；三维骨架为单目相对估计。"
        if count > 1:
            focus += "当前跌倒分数属于整个窗口，不能定位到某个人。"
        elif count == 0:
            focus += "未检测出姿态不等于无人或安全。"
        for person in detection.get('people', [])[:4]:
            individual = person.get('analysis') or {}
            observation = person.get('observation') or {}
            label = {'adl':'正常活动特征','model_disagreement':'姿态模型疑似跌倒，RGB未确认，需复核','fall_candidate':'疑似跌倒特征','fall_sustained':'连续跌倒特征，需复核','partial_adl':'局部画面正常活动特征','partial_fall_candidate':'局部画面疑似跌倒，需复核','unobservable':'缺少有效人物图像','warming_up':'建立个人窗口'}.get(individual.get('state'), '等待判断')
            activity = individual.get('activity') or {}
            if activity:
                names = {'walk':'行走','fall':'跌倒过程','fallen':'跌倒后倒地','sit_down':'坐下','sitting':'坐姿','lie_down':'主动躺下','lying':'躺姿（不等于睡眠）','stand_up':'起身','standing':'站姿','other':'其他活动'}
                label += f"；动作模型候选{names.get(activity.get('label'),'未知')}，分数{activity.get('confidence',0):.2f}"
            focus += f"人物{person.get('track_id', person.get('index'))}：{label}（{observation.get('reason', '未知')}）。"
        if any(p.get('analysis', {}).get('scope') == 'tracked_person_crop' for p in detection.get('people', [])):
            focus += '个人裁剪分数尚未完成事件级验证。'
    clip = video.get("clip_classification") or {}
    if clip:
        label = "跌倒片段" if clip.get("predicted") == "fall" else "正常活动片段"
        focus += f"整段回放视频模型分类为{label}，未定位发生时间，该结果不是当前现场告警。"
    return (
        f"过去 {context['hours']} 小时共记录 {alerts} 条预警，其中 {unhandled} 条未处理；"
        f"生命体征样本 {vitals} 条，行为样本 {actions} 条（当前采集源 {context.get('vitals_source', '未知')}）。{focus} "
        "毫米波模型目前处于独立采集/特征阶段，摘要不会把未验证雷达结果当作诊断结论。"
    )


def summarize(context: dict) -> tuple[str, str, str | None]:
    settings = config()
    with _runtime_lock:
        runtime = dict(_runtime)
    key = runtime.get("api_key") if "api_key" in runtime else (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not key:
        return local_summary(context), "local", None

    prompt = (
        "你是养老机构的值班分析助手。请基于下面的结构化监测数据，用简洁中文输出三段："
        "当前概况、需要优先处理的事项、下一步建议。不得编造不存在的数值，不做医学诊断；"
        "若是 mock 要说明模拟；file_replay 是回放测试，不能要求按现场报警处理。"
        "R3D 仅训练 ADL/Fall 两类；活动帧差、特征变化和 CAM 为辅助证据，不能推断心率、睡眠、医学诊断。"
        "模型分数不是校准风险概率，过期结果不能代表当前状态。\n\n"
        "人物编号为短期跟踪编号，不是身份；三维姿态是相对估计，模拟点云不是雷达实测，场景跌倒分数不能归到个人。"
        "各人物analysis为独立裁剪片段结果，validated=false表示个人场景尚未验证，几何姿态不属于训练动作类别，不能将局部观测当成正常或跌倒。\n\n"
        + json.dumps(context, ensure_ascii=False)
    )
    payload = {
        "model": settings["model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你负责把监测数据整理成可执行的值班摘要。"},
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        f"{settings['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(os.getenv("LLM_TIMEOUT", "20"))) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not str(content).strip():
            raise ValueError("empty model response")
        return str(content).strip(), "openai-compatible", None
    except (OSError, URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return local_summary(context), "local-fallback", f"{type(exc).__name__}: {exc}"


def now_ms() -> int:
    return int(time.time() * 1000)

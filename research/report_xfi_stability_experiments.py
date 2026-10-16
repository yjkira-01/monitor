"""Paired optimizer comparison after independent saved-model verification."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from report_xfi_accuracy_experiments import CONDITIONS, SCORE, VIEWS, NAMES, compare, label


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(root, count):
    report = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / "analysis/verification.json").read_text(encoding="utf-8"))
    if (report["status"] != "complete" or receipt["status"] != "complete" or len(receipt["models"]) != count or
            receipt["run_summary_sha256"] != sha(root / "run_summary.json") or report["final_test_subjects_used"]):
        raise ValueError("Complete independently verified development models required")
    return report, receipt


def record(report, method, posthoc=False):
    subjects = report["validation_subjects"]
    value_key = "prediction_recentered_MPJPE_mm" if posthoc else "MPJPE_mm"
    subject_key = "prediction_recentered_per_subject_mm" if posthoc else "per_validation_subject_MPJPE_mm"
    conditions = {}
    scores = []
    for c in CONDITIONS:
        rows = [run["models"][method]["metrics"][c] for run in report["runs"]]
        values = np.asarray([row[value_key] for row in rows])
        conditions[c] = {"MPJPE_mean_mm": float(values.mean()), "initialization_sd_mm": float(values.std(ddof=1)),
                         "per_initialization_mm": values.tolist(),
                         "per_subject_initialization_mean_mm": {str(s): float(np.mean([row[subject_key][str(s)] for row in rows])) for s in subjects}}
    for run in report["runs"]:
        scores.append([float(np.mean([run["models"][method]["metrics"][c][subject_key][str(s)] for c in SCORE])) for s in subjects])
    scores = np.asarray(scores)
    return {"conditions": conditions, "subject_scores_mm": scores.tolist(), "per_initialization_score_mm": scores.mean(1).tolist(),
            "score_mean_mm": float(scores.mean()), "score_initialization_sd_mm": float(scores.mean(1).std(ddof=1))}


def variation(root, report):
    result = []
    for run in report["runs"]:
        for method in report["methods"]:
            path = root / f"seed{run['seed']}" / f"{method}_predictions.npz"
            if sha(path) != run["models"][method]["predictions_sha256"]:
                raise ValueError("Prediction source changed")
            with np.load(path, allow_pickle=False) as saved:
                row = {"seed": run["seed"], "method": method}
                for key in ("target_mm", "clean_fused_prediction_mm"):
                    array = saved[key]
                    row[key + "_frame_deviation_rms_mm"] = float(np.sqrt(np.square(array - array.mean(0)).mean()))
                target = saved["target_mm"].reshape(-1, 17, 3)
                for branch in ("fused", "video", "radar"):
                    prediction = saved["clean_" + branch + "_prediction_mm"].reshape(-1, 17, 3)
                    row["clean_" + branch + "_MPJPE_mm"] = float(np.linalg.norm(prediction - target, axis=-1).mean())
                row["clean_fusion_gain_vs_best_same_model_single_modality_mm"] = min(row["clean_video_MPJPE_mm"], row["clean_radar_MPJPE_mm"]) - row["clean_fused_MPJPE_mm"]
                result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("control", type=Path)
    args = parser.parse_args()
    baseline, control = args.baseline.resolve(), args.control.resolve()
    base, base_audit = verified(baseline, 9)
    low, low_audit = verified(control, 3)
    if base["training"]["lr"] != .001 or low["training"]["lr"] != .0001 or low["methods"] != ["native_mse"]:
        raise ValueError("Prespecified learning-rate comparison required")
    for key in ("cache_report_sha256", "official_source_sha256", "initialization_seeds", "train_subjects", "validation_subjects", "train_frames", "validation_frames", "protocol_sha256"):
        if base[key] != low[key]:
            raise ValueError("Optimizer controls differ in " + key)
    for key in base["training"]:
        if key != "lr" and base["training"][key] != low["training"][key]:
            raise ValueError("Additional training change: " + key)
    pairing = []
    for a, b in zip(base["runs"], low["runs"]):
        if a["seed"] != b["seed"]:
            raise ValueError("Initialization order differs")
        left, right = a["models"]["native_mse"], b["models"]["native_mse"]
        for key in ("initial_state_sha256", "schedule_sha256", "parameters", "frozen_backbone_parameters_and_buffers_unchanged"):
            if left[key] != right[key]:
                raise ValueError("Paired initialization/budget differs: " + key)
        pairing.append({"seed": a["seed"], "initial_state_sha256": left["initial_state_sha256"], "schedule_sha256": left["schedule_sha256"]})
    records = {label(v): record(base, v[0], v[1] == "recentered") for v in VIEWS}
    records.update({"low_lr:fused": record(low, "native_mse"), "low_lr:recentered": record(low, "native_mse", True)})
    names = {label(v): NAMES[v] for v in VIEWS}
    names.update({"low_lr:fused": "MSE lr0.0001", "low_lr:recentered": "MSE lr0.0001＋预测髋重中心"})
    strongest = min((label(v) for v in VIEWS), key=lambda k: records[k]["score_mean_mm"])
    comparisons = {"primary_lr_control": compare("low_lr:fused", "native_mse:fused", records, base["validation_subjects"], base["initialization_seeds"]),
                   "posthoc_lr_control": compare("low_lr:recentered", "native_mse:recentered", records, base["validation_subjects"], base["initialization_seeds"]),
                   "development_strongest_original_control": compare("low_lr:fused", strongest, records, base["validation_subjects"], base["initialization_seeds"])}
    output = control / "reporting"
    if output.exists():
        raise ValueError("Refusing to overwrite paired optimizer report")
    result = {"scope": "Exploratory one-split paired fixed-learning-rate comparison; no novelty or final-test claim",
              "source_sha256": {"baseline_summary": sha(baseline / "run_summary.json"), "control_summary": sha(control / "run_summary.json"),
                                "baseline_audit": sha(baseline / "analysis/verification.json"), "control_audit": sha(control / "analysis/verification.json"), "reporter": sha(__file__)},
              "pairing": pairing, "records": records, "comparisons": comparisons,
              "prediction_variation": {"original": variation(baseline, base), "low_learning_rate": variation(control, low)},
              "final_test_subjects_used": []}
    lines = ["# X-Fi mRI适配：固定低学习率配对结果", "",
             "原三目标×三初始化九套100轮训练与追加三套低学习率MSE训练均已完成、独立核验。追加组仅lr0.001→0.0001，原生结构、冻结骨干、初始化、批次/扰动日程、损失、优化器其余参数和末轮评价一致。", "",
             "训练12主体1152帧、开发验证4主体384帧、初始化7/17/27；一个主体划分，最终主体3/10/12/18未预测。±为三初始化样本标准差，不是置信区间。所有误差单位mm，越低越好。", "",
             "| 方法 | 正常MPJPE | 五条件S | 暗光 | 噪声暗光 | 遮挡 | 缺视频 | 缺雷达 | 稀疏雷达 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in records.items():
        c = row["conditions"]["clean"]
        values = [f"{c['MPJPE_mean_mm']:.3f}±{c['initialization_sd_mm']:.3f}",
                  f"{row['score_mean_mm']:.3f}±{row['score_initialization_sd_mm']:.3f}"]
        values += [f"{row['conditions'][c]['MPJPE_mean_mm']:.3f}" for c in CONDITIONS[1:]]
        lines.append("| " + names[key] + " | " + " | ".join(values) + " |")
    lines += ["", "## 配对收益及主体方向", "",
              "S改善为正表示误差减少，正常误差变化为正表示变差。门槛为开发投入判据，不是显著性检验。对原阶段最强控制的比较按开发S择优，属于探索性比较。", ""]
    for key, row in comparisons.items():
        lines += [f"- {row['candidate']} 对 {row['reference']}：S改善{row['score_gain_mm']:.3f}mm（{row['score_gain_percent']:.2f}%），正常误差变化{row['clean_change_mm']:+.3f}mm；资源门槛{'通过' if row['resource_gate_passed'] else '未通过'}。",
                  f"  四主体改善：{row['per_subject_gain_mm']}；三个初始化改善：{row['per_initialization_gain_mm']}。"]
    lines += ["", "## 完整性与结论范围", "",
              "三初始化的初始state SHA、批次及扰动日程SHA逐项一致；两阶段共享同一完整缓存、官方源码、骨干、训练/验证目标和参数量。12套权重与保存预测均经独立CUDA重建、独立NumPy指标及主体宏分数复算；双缺失先验和单缺失回退、原图与点云端到端重建均通过。", "",
              f"原阶段最大预测重建差{base_audit['max_prediction_difference_mm']:.9g}mm，追加阶段{low_audit['max_prediction_difference_mm']:.9g}mm；原阶段指标复算差{base_audit['max_metric_difference_mm']:.9g}mm，追加阶段{low_audit['max_metric_difference_mm']:.9g}mm。此前完整CPU/GPU重建0.3404mm未达0.01mm容差的限制继续保留，不放宽容差。", "",
              "预测跨帧变化诊断保留在paired_accuracy_report.json。变化增大本身不等于准确度提升，只用于识别近乎固定输出，实际收益由MPJPE判定。", "",
              "同一诊断同时保存每个模型正常画面的融合、RGB-only、radar-only误差及融合对较强单路的收益。这些是同一随机模态训练模型的三种推理子集，不是另行训练的独立单路模型。", "",
              "低学习率是常规优化；不能把改善归因于新的融合机制，也不声称创新。X-Fi原MM-Fi论文88.8mm不是同数据/协议对照。此前RTMO流程65.86mm属于不同输入与预训练流程；本阶段赢较弱X-Fi适配并不自动意味着提升项目已知最好准确度。人工扰动、视觉生成标签、小样本及单主体划分限制泛化判断，后续需扩充样本和独立主体划分。", "",
              "[追加固定协议](../../../research/xfi_stability_protocol_20261013.md)；[本组原始摘要](../run_summary.json)；[本组独立核验](../analysis/verification.json)。", ""]
    output.mkdir()
    (output / "paired_accuracy_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "paired_accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "reporter_source.py").write_bytes(Path(__file__).read_bytes())
    csv_rows = []
    for key, row in records.items():
        for index, seed in enumerate(base["initialization_seeds"]):
            csv_rows.append({"view": key, "seed": seed, "score_mm": row["per_initialization_score_mm"][index],
                             **{c + "_MPJPE_mm": row["conditions"][c]["per_initialization_mm"][index] for c in CONDITIONS}})
    with (output / "paired_metrics_by_initialization.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": "complete", "comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()

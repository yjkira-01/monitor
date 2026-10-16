"""Report verified development results and preserve ordinary posthoc controls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("clean", "dark", "noisy_dark", "occlusion", "video_missing", "radar_missing", "radar_sparse")
SCORE = ("clean", "noisy_dark", "occlusion", "video_missing", "radar_missing")
VIEWS = (("native_mse", "fused"), ("native_mse", "recentered"),
         ("native_mpjpe", "fused"), ("native_mpjpe", "recentered"), ("root_mpjpe", "fused"))
NAMES = {("native_mse", "fused"): "X-Fi适配：MSE",
         ("native_mse", "recentered"): "MSE＋预测髋重中心",
         ("native_mpjpe", "fused"): "MPJPE损失",
         ("native_mpjpe", "recentered"): "MPJPE损失＋预测髋重中心",
         ("root_mpjpe", "fused"): "训练及推理髋中心约束"}
JOINTS = ("鼻", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
          "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label(view):
    return ":".join(view)


def compare(candidate, reference, records, subjects, seeds):
    a, b = records[candidate], records[reference]
    gains = np.asarray(b["subject_scores_mm"]) - np.asarray(a["subject_scores_mm"])
    gain = float(gains.mean())
    clean_change = a["conditions"]["clean"]["MPJPE_mean_mm"] - b["conditions"]["clean"]["MPJPE_mean_mm"]
    threshold = max(1., .02 * b["score_mean_mm"])
    clean_limit = max(1., .02 * b["conditions"]["clean"]["MPJPE_mean_mm"])
    return {"candidate": candidate, "reference": reference, "score_gain_mm": gain,
            "score_gain_percent": gain / b["score_mean_mm"] * 100,
            "clean_change_mm": clean_change, "score_threshold_mm": threshold, "clean_limit_mm": clean_limit,
            "per_subject_gain_mm": {str(s): float(v) for s, v in zip(subjects, gains.mean(0))},
            "per_initialization_gain_mm": {str(s): float(v) for s, v in zip(seeds, gains.mean(1))},
            "resource_gate_passed": bool(gain >= threshold and clean_change <= clean_limit and
                                          np.count_nonzero(gains.mean(0) > 0) >= 3 and
                                          np.count_nonzero(gains.mean(1) > 0) >= 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    root = args.experiment.resolve()
    report_path = root / "run_summary.json"
    verification_path = root / "analysis/verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (report["status"] != "complete" or verification["status"] != "complete" or
            verification["run_summary_sha256"] != digest(report_path) or
            report["initialization_seeds"] != [7, 17, 27] or report["training"]["epochs"] != 100 or
            len(verification["models"]) != 9 or report["final_test_subjects_used"]):
        raise ValueError("Only independently verified complete nine-model formal development results may be reported")
    output = root / "reporting"
    if output.exists():
        raise ValueError("Refusing to overwrite an existing report")
    subjects, seeds = report["validation_subjects"], report["initialization_seeds"]
    with np.load(Path(report["cache"]) / "calibrated_backbones.npz", allow_pickle=False) as arrays:
        training_prior = arrays["training_prior_m"].astype(np.float64) * 1000
    prior_subject_errors = None
    records, csv_rows = {}, []
    for view in VIEWS:
        method, variant = view
        per_condition = {c: [] for c in CONDITIONS}
        score_subjects, branch_gains = [], []
        for run in report["runs"]:
            path = root / f"seed{run['seed']}" / f"{method}_predictions.npz"
            if digest(path) != run["models"][method]["predictions_sha256"]:
                raise ValueError("Prediction source changed after verification")
            with np.load(path, allow_pickle=False) as saved:
                target = saved["target_mm"].reshape(-1, 17, 3)
                identifiers = saved["subject_ids"]
                if prior_subject_errors is None:
                    prior_error = np.linalg.norm(target - training_prior, axis=-1)
                    prior_subject_errors = {str(s): float(prior_error[identifiers == s].mean()) for s in subjects}
                condition_subject = {}
                for condition in CONDITIONS:
                    prediction = saved[condition + "_" + variant + "_prediction_mm"].reshape(-1, 17, 3)
                    error = np.linalg.norm(prediction - target, axis=-1)
                    subject_error = [float(error[identifiers == s].mean()) for s in subjects]
                    condition_subject[condition] = subject_error
                    row = {"mean_mm": float(np.mean(subject_error)), "frame_p95_mm": float(np.percentile(error.mean(1), 95)),
                           "per_joint_mm": error.mean(0).tolist(), "per_subject_mm": subject_error}
                    per_condition[condition].append(row)
                    csv_rows.append({"view": label(view), "seed": run["seed"], "condition": condition,
                                     "MPJPE_mm": row["mean_mm"], "frame_p95_mm": row["frame_p95_mm"],
                                     **{f"subject{s}_MPJPE_mm": value for s, value in zip(subjects, subject_error)}})
                score_subjects.append(np.mean([condition_subject[c] for c in SCORE], axis=0).tolist())
                branch_errors = []
                for branch in ("video", "radar"):
                    prediction = saved["clean_" + branch + "_prediction_mm"].reshape(-1, 17, 3)
                    if variant == "recentered":
                        prediction = prediction - ((prediction[:, 11] + prediction[:, 12]) / 2)[:, None]
                    branch_errors.append(float(np.linalg.norm(prediction - target, axis=-1).mean()))
                branch_gains.append(min(branch_errors) - per_condition["clean"][-1]["mean_mm"])
        subject_scores = np.asarray(score_subjects)
        records[label(view)] = {"name": NAMES[view], "score_conditions": SCORE,
                "score_mean_mm": float(subject_scores.mean()), "score_initialization_sd_mm": float(subject_scores.mean(1).std(ddof=1)),
                "per_initialization_score_mm": subject_scores.mean(1).tolist(), "subject_scores_mm": score_subjects,
                "clean_gain_vs_same_model_best_single_modality_mm": branch_gains,
                "conditions": {c: {"MPJPE_mean_mm": float(np.mean([x["mean_mm"] for x in rows])),
                                   "initialization_sd_mm": float(np.std([x["mean_mm"] for x in rows], ddof=1)),
                                   "mean_of_initialization_frame_p95_mm": float(np.mean([x["frame_p95_mm"] for x in rows])),
                                   "per_joint_initialization_mean_mm": np.mean([x["per_joint_mm"] for x in rows], axis=0).tolist(),
                                   "per_subject_initialization_mean_mm": np.mean([x["per_subject_mm"] for x in rows], axis=0).tolist()}
                               for c, rows in per_condition.items()}}
    best_posthoc = min(("native_mse:recentered", "native_mpjpe:recentered"), key=lambda k: records[k]["score_mean_mm"])
    comparisons = {"mpjpe_vs_mse": compare("native_mpjpe:fused", "native_mse:fused", records, subjects, seeds),
                   "root_vs_mse": compare("root_mpjpe:fused", "native_mse:fused", records, subjects, seeds),
                   "root_vs_best_posthoc": compare("root_mpjpe:fused", best_posthoc, records, subjects, seeds)}
    independent = comparisons["root_vs_best_posthoc"]
    published = report["aggregate"]["root_training_vs_best_posthoc_control"]
    if (abs(independent["score_gain_mm"] - published["score_gain_mm"]) > 1e-8 or
            independent["resource_gate_passed"] != published["resource_gate_passed"]):
        raise ValueError("Reporting comparison differs from audited training summary")
    result = {"scope": "One development subject split, three initializations; ordinary optimization, no novelty or final-test claim",
              "source_run_sha256": digest(report_path), "verification_sha256": digest(verification_path),
              "reporter_sha256": digest(__file__), "records": records, "comparisons": comparisons,
              "training_mean_pose_without_observations": {"scope": "Supplementary descriptive check, no additional fitting or selection",
                    "MPJPE_mm": float(np.mean(list(prior_subject_errors.values()))), "per_subject_mm": prior_subject_errors},
              "final_test_subjects_used": []}
    lines = ["# X-Fi原生模块mRI适配：准确度优化结果", "",
             "九套正式开发训练及独立核验已完成。训练12主体1152帧，开发验证4主体384帧；三个初始化7、17、27共享一个主体划分。最终测试主体3、10、12、18未预测。", "",
             "保留原X-Fi骨干、512维/32tokens/8头/4层融合、五槽索引和31,806,675参数。预训练学习参数与训练观测校准后的BN均冻结；三方法共享原生tokens、初始化、增强与模态子集日程。每套100轮、batch16、AdamW lr0.001/wd0.01，取末轮。它是mRI适配，未复现MM-Fi原论文协议。", "",
             "所有数值单位mm，越低越好。±为三初始化样本标准差，不是置信区间。S为五预定条件的主体宏平均；dark与radar_sparse不进入S。主指标相对髋中心目标；后处理两行明确减去预测髋中点。", "",
             "| 方法 | 正常MPJPE | 五条件S | 暗光 | 噪声暗光 | 遮挡 | 缺视频 | 缺雷达 | 稀疏雷达 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for view in VIEWS:
        row = records[label(view)]
        clean = row["conditions"]["clean"]
        values = [f"{clean['MPJPE_mean_mm']:.3f}±{clean['initialization_sd_mm']:.3f}",
                  f"{row['score_mean_mm']:.3f}±{row['score_initialization_sd_mm']:.3f}"]
        values += [f"{row['conditions'][c]['MPJPE_mean_mm']:.3f}" for c in CONDITIONS[1:]]
        lines.append("| " + row["name"] + " | " + " | ".join(values) + " |")
    lines += ["", "## 配对判据", "",
              "正的S改善表示候选误差更低；正的正常误差变化表示正常画面变差。门槛只决定继续投入，不是统计显著性判断。", ""]
    for key, row in comparisons.items():
        lines += [f"- {row['candidate']} 对 {row['reference']}：S改善{row['score_gain_mm']:.3f}mm（{row['score_gain_percent']:.2f}%），正常误差变化{row['clean_change_mm']:+.3f}mm；资源门槛{'通过' if row['resource_gate_passed'] else '未通过'}。",
                  f"  四主体改善：{row['per_subject_gain_mm']}；三个初始化改善：{row['per_initialization_gain_mm']}。"]
    lines += ["", "## 误差定位及证据", ""]
    for view in VIEWS:
        row = records[label(view)]
        values = row["conditions"]["clean"]["per_joint_initialization_mean_mm"]
        largest = sorted(range(17), key=lambda j: values[j], reverse=True)[:4]
        lines.append(f"- {row['name']}：正常画面P95的三初始化均值{row['conditions']['clean']['mean_of_initialization_frame_p95_mm']:.3f}mm；最大四个关节为" + "、".join(f"{JOINTS[j]} {values[j]:.2f}mm" for j in largest) + "。")
    lines += ["", f"另作描述性检查：仅输出训练目标均值、完全不读取两路观测的误差为{np.mean(list(prior_subject_errors.values())):.3f}mm，四主体为{prior_subject_errors}；它帮助辨别模型是否学到了有效观测证据，不替代论文主基线或资源判据。", "",
              "分支诊断使用同一随机模态训练模型的RGB-only/radar-only推理，并非独立训练单路模型。完整逐主体、逐关节、逐初始化值及正常融合相对较强单路的收益保留在accuracy_report.json和metrics_by_initialization.csv。", "",
              f"保存权重同GPU预测核验最大差{verification['max_prediction_difference_mm']:.9g}mm，独立指标最大差{verification['max_metric_difference_mm']:.9g}mm；主体宏分数、缺失回退、冻结参数与原图/雷达重建核验通过。完整CPU/GPU重建此前0.3404mm未达0.01mm容差的限制继续保留，未放宽容差。", "",
              "常规MPJPE损失、髋中心约束与预测重中心均有先例，本阶段不声称原创。原X-Fi MM-Fi I+R 88.8mm与本实验不是相同数据/协议；旧RTMO小模型65.86mm又是不同输入与预训练流程，跨流程差值不归因于融合模块。人工扰动、视觉生成标签、较小开发样本和单主体划分限制结论；需要扩大训练样本及独立划分后再讨论泛化，最终测试继续封存。", "",
              "协议：[固定准确度对照协议](../../../research/xfi_accuracy_protocol_20261013.md)；原始摘要：[run_summary.json](../run_summary.json)；独立核验：[verification.json](../analysis/verification.json)。", ""]
    output.mkdir()
    (output / "accuracy_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "reporter_source.py").write_bytes(Path(__file__).read_bytes())
    with (output / "metrics_by_initialization.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": "complete", "best_posthoc": best_posthoc, "comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()

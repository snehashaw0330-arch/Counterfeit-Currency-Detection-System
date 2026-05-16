"""Walk the diagnostic_report.json and surface failure patterns."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "tests", "diagnostic_report.json")

with open(REPORT, encoding="utf-8") as f:
    data = json.load(f)

print("=" * 90)
print("1. OCR RAW OUTPUT ON REAL NOTES (to see what Tesseract actually reads)")
print("=" * 90)

for row in data:
    if row["label"] != "real":
        continue

    fa = row["forensic_analysis"]["ocr_serial_number"]
    diag = row["diagnostic"]

    print(f"\n--- {row['file']}  (verdict: {fa['status']})")
    print(f"  detected: {fa.get('value')!r}")
    print(f"  top-left raw    : {diag['ocr_top_left_raw'][:160]!r}")
    print(f"  bottom-right raw: {diag['ocr_bottom_right_raw'][:160]!r}")


print("\n" + "=" * 90)
print("2. EVERY FORENSIC CHECK — PASS/FAIL COUNT PER CHECK")
print("=" * 90)

checks = [
    "uv_light_detection",
    "watermark_detection",
    "ocr_serial_number",
    "gandhi_face_analysis",
    "security_thread_detection",
    "hologram_detection",
    "denomination_classification",
]

for check in checks:

    real_pass = sum(
        1 for r in data
        if r["label"] == "real"
        and r["forensic_analysis"][check]["status"] == "PASS"
    )
    real_fail = sum(
        1 for r in data
        if r["label"] == "real"
        and r["forensic_analysis"][check]["status"] == "FAIL"
    )
    fake_pass = sum(
        1 for r in data
        if r["label"] == "fake"
        and r["forensic_analysis"][check]["status"] == "PASS"
    )
    fake_fail = sum(
        1 for r in data
        if r["label"] == "fake"
        and r["forensic_analysis"][check]["status"] == "FAIL"
    )

    sep_score = real_pass + fake_fail
    print(
        f"  {check:<32}  real PASS {real_pass}/10  FAIL {real_fail}/10  "
        f"|  fake PASS {fake_pass}/10  FAIL {fake_fail}/10  "
        f"|  separator score {sep_score}/20"
    )

print("\n" + "=" * 90)
print("3. SIGNAL DISTRIBUTION (raw numbers per class)")
print("=" * 90)

def stats(values):
    if not values:
        return "n/a"
    vs = sorted(values)
    return (
        f"min {vs[0]:8.1f}  "
        f"med {vs[len(vs) // 2]:8.1f}  "
        f"max {vs[-1]:8.1f}"
    )

for key in (
    "wm_variance",
    "uv_ratio_pct",
    "thread_vert_lines",
    "hologram_score",
):
    real = [r["diagnostic"][key] for r in data if r["label"] == "real"]
    fake = [r["diagnostic"][key] for r in data if r["label"] == "fake"]

    print(f"\n  {key}")
    print(f"    real: {stats(real)}")
    print(f"    fake: {stats(fake)}")


print("\n" + "=" * 90)
print("4. MISCLASSIFICATIONS")
print("=" * 90)

print("\n  REAL notes flagged SUSPICIOUS:")
for r in data:
    if r["label"] == "real" and r["final_verdict"] == "SUSPICIOUS":
        print(
            f"    {r['file']:<30} raw={r['model_raw']*100:5.1f}% "
            f"forensic={r['forensic_pass']}/{r['forensic_total']} "
            f"combined={r['combined_score']*100:5.1f}%"
        )

print("\n  FAKE samples wrongly approved as REAL:")
for r in data:
    if r["label"] == "fake" and r["final_verdict"] == "REAL":
        print(
            f"    {r['file']:<30} raw={r['model_raw']*100:5.1f}% "
            f"forensic={r['forensic_pass']}/{r['forensic_total']} "
            f"combined={r['combined_score']*100:5.1f}%"
        )

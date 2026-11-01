"""Exercise real trained weights on an image and a generated video fixture.

The fixture uses held-out dry images; it is not footage from a physical camera.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "outputs"
output.mkdir(exist_ok=True)
with (ROOT / "dataset/manifest.csv").open(encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
cracked = [row for row in rows if row["label"] == "crack"][:2]
uncracked = [row for row in rows if row["label"] == "no_crack"][:2]
selected = [cracked[0], uncracked[0], cracked[1], uncracked[1]]
images = [cv2.imread(row["prepared"]) for row in selected]
frame = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])
fixture = output / "heldout_mosaic.jpg"
assert cv2.imwrite(str(fixture), frame)
(output / "fixture_sources.json").write_text(json.dumps({
    "purpose": "Inference software smoke test, not localization validation or underwater footage",
    "tiles": selected,
}, indent=2), encoding="utf-8")
video = output / "smoke_fixture.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (512, 512))
assert writer.isOpened()
for _ in range(4):
    writer.write(frame)
writer.release()
for arguments in (
    ["image", "--source", str(fixture), "--output", str(output / "heldout_candidates.jpg")],
    ["live", "--source", str(video), "--output", str(output / "smoke_annotated.mp4"), "--no-display", "--max-frames", "4"],
):
    subprocess.run([sys.executable, str(ROOT / "crack_vision.py"), *arguments], cwd=ROOT, check=True)
metadata = json.loads((output / "heldout_candidates.json").read_text())
assert metadata["tiles"] == 9
logs = [json.loads(line) for line in (output / "smoke_annotated.jsonl").read_text().splitlines()]
assert len(logs) == 4
capture = cv2.VideoCapture(str(output / "smoke_annotated.mp4"))
count = 0
while True:
    ok, decoded = capture.read()
    if not ok:
        break
    assert decoded.shape[:2] == (512, 512)
    count += 1
capture.release()
assert count == 4
print("Real-checkpoint image inference and 4-frame video round trip passed.")

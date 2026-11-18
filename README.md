# Underwater dam inspection: crack vision prototype

This implements the AI phase of the supplied project report: a **YOLO11n-cls**
crack/no-crack classifier trained on SDNET2018 wall images, with evaluation,
image analysis, and a camera/video display of approximate crack candidates.

The classifier uses dry concrete photographs. It is not yet validated on
underwater footage. Its sliding-window boxes are coarse candidate regions,
not supervised detection boxes, crack boundaries, physical crack counts,
crack depth measurements, or a dam-health assessment.

## Environment

Use Python 3.13 and the project environment from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

For CPU use, select the CPU package index instead. `requirements-lock.txt`
records the actual environment used for this experiment after installation.
The NVIDIA driver, rather than a separately installed CUDA toolkit, supports
the CUDA runtime supplied with the PyTorch wheels.

## Dataset preparation

The raw dataset is already extracted in this folder. No ZIP extraction is needed.

```powershell
.\.venv\Scripts\python.exe crack_vision.py prepare
```

Preparation decodes every wall image, verifies RGB 256×256 dimensions, and
computes SHA-256 hashes. Exact duplicates with the same label and source
photograph are deduplicated. Images with conflicting duplicate labels, and
duplicates crossing source groups, are excluded entirely and documented in
`dataset/duplicate_audit.json`. Raw images are preserved.

The prefix before the hyphen in each filename identifies the inferred source
photograph. All patches from a photograph stay in one split; the wall folder
contains 72 such groups, consistent with the dataset readme. A fixed seed
selects approximately 70%/15%/15% of groups for train/validation/test, balancing
patch class counts using only labels. No model results influence the split.
This reduces photograph leakage; it does not demonstrate generalization to
independent buildings, dams, or underwater environments.

`dataset/manifest.csv` records each image, hash, label, group, and split.
`dataset/summary.json` records counts and the manifest hash. Prepared files
are hardlinks when supported, with copying as a fallback. Treat prepared
images as read-only because hardlinks share the original file's contents.
Preparation rejects a nonempty output directory. To try other surfaces use
a fresh location:

```powershell
.\.venv\Scripts\python.exe crack_vision.py prepare --surfaces W D P --dataset dataset_all
```

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe crack_vision.py train --epochs 20 --batch 24
```

Training downloads the official pretrained `yolo11n-cls.pt`, uses 224×224
inputs and AdamW, and saves the best validation top-1 checkpoint. Full-patch
resize, horizontal/vertical flips and mild color jitter retain the entire
patch. Random crops and erasing are disabled because they can remove the
crack that determines the image label. Class frequencies remain natural;
the validation threshold addresses the precision/recall tradeoff, not the
training imbalance itself.

After training, evaluation selects a threshold maximizing **validation F2**,
which weights recall more strongly than precision. This is an experimental
operating point; high recall is not guaranteed. The held-out test set is then
evaluated at both that threshold and 0.5, without tuning on test predictions.
Scores are model outputs and have not been independently calibrated as
real-world probabilities. Threshold behavior on merged sliding-window maps
is also unvalidated because the dataset has no localization annotations.

Artifacts:

- `models/best.pt`, `models/last.pt`: actual trained checkpoints.
- `models/training.json`: hardware, preprocessing, timing and run location.
- `models/metrics.json`: validation/test accuracy, precision, recall, F1,
  F2, confusion matrices, ROC AUC, average precision and inference throughput.
- `models/val_predictions.csv`, `models/test_predictions.csv`: individual scores.
- `models/evaluation.png`: test confusion matrix and precision–recall curve.
- `runs/wall_yolo11n/`: epoch records, configuration and training plots.

Inference timings exclude file decoding and warmup, and measure batches;
they should not be interpreted as camera FPS. The first image command includes
model inference initialization in its processing time.

Training, validation and inference resize with the same PIL/torchvision
operation. Inference supplies RGB float tensors to bypass the library's
default cropping transforms. The preprocessing consistency test compares
these tensors exactly.

Re-evaluate existing weights:

```powershell
.\.venv\Scripts\python.exe crack_vision.py evaluate
```

## Image and live video

```powershell
.\.venv\Scripts\python.exe crack_vision.py image --source path\to\photo.jpg --output outputs\result.jpg
.\.venv\Scripts\python.exe crack_vision.py live --source 0
.\.venv\Scripts\python.exe crack_vision.py live --source inspection.mp4 --output outputs\inspection.mp4
```

The image command saves an annotated image, a color score map, and JSON with
candidate boxes and settings. The video command can save an annotated MP4 and
a JSONL record per frame. Press Q or Escape to exit the display. Device index
0 is the default USB camera; try 1 if another camera occupies that index.
File processing can run without a display using `--no-display`.

The software checks and a real-checkpoint image/video smoke test can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\smoke_inference.py
```

The smoke test makes a four-frame video from held-out dry patches and verifies
that the saved annotated video and inspection logs can be read. It does not
test a physical camera, underwater imagery, or localization accuracy.

Default windows are 256×256 with a 128-pixel stride. Edge windows cover the
entire frame. Overlapping scores are averaged and thresholded, then connected
regions receive boxes. One crack can create multiple boxes; multiple cracks
can merge. Camera frames are limited to 640 pixels wide by default to reduce
processing load; set `--max-width 0` to keep the original resolution. Resizing
can remove fine-crack detail. Logs use coordinates in the processed frame.
For recorded files, saved playback FPS follows the source FPS; processing can
take longer than real time. A camera run records frames as processed and may
drop frames, so saved playback timing is not a precise acquisition clock.

The tuned threshold is loaded only if its stored weight hash matches the
checkpoint. `--threshold 0.5` supplies a manual override. `--device cpu` forces
CPU inference. The live loop processes every captured frame sequentially;
real-time feasibility must be measured with the intended camera.

## Raspberry Pi with camera and ultrasonic sensor

On Raspberry Pi OS (64-bit), install the system packages first and create the
environment with access to them:

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-lgpio
python3 -m venv --system-site-packages venv && source venv/bin/activate
pip install torch torchvision ultralytics
```

`models/best.pt` is not in git (see `.gitignore`); copy it to the Pi with `scp`.

```bash
python crack_vision.py live --source picamera --smooth
python crack_vision.py live --source picamera --smooth --ultrasonic --hfov 66
```

`--source picamera` reads the Pi Camera Module. `--smooth` runs detection in a
background thread so the video stays fluid and boxes update when ready.
`--ultrasonic` reads a trigger/echo sensor (default BCM pins `--trig 23
--echo 24`, wiring in `docs/WIRING.md`), shows the wall distance on the video,
and adds `distance_m`, `mm_per_pixel` and approximate `width_mm`/`height_mm`
to each logged box. The sensor measures distance to the wall, not crack depth.
Millimetre sizes assume the camera faces the wall squarely and `--hfov` is the
camera's horizontal field of view in the medium. Pass `--sound-speed 1480` for
a sensor operating in fresh water.

### Headless operation (no monitor)

Add `--no-display --stream 8000` and open `http://<pi-address>:8000/` in a
browser on any computer on the same network. To start at boot, edit
`deploy/crack-vision.service` (user and paths), then on the Pi:

```bash
sudo cp deploy/crack-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crack-vision
journalctl -u crack-vision -f      # live log; use systemctl stop crack-vision to stop
```

Each run saves `outputs/run_<date>.mp4` and a matching `.jsonl` log. The stream
has no password, so use it only on a trusted network or tether.

## Underwater validation required next

Collect cracked and uncracked specimens in freshwater with the actual camera.
Record specimen ID, distance, lighting, turbidity and motion conditions.
Keep all frames from one specimen/session together when splitting data.
Measure missed cracks, false alarms and camera processing rate before claiming
underwater capability. Pixel labels or verified bounding boxes are needed to
train and evaluate precise localization. No physical camera or underwater
experiment can be validated from SDNET2018 alone.

## Sources

- Dataset documentation: `DATA_Maguire_20180517_ALL/DATA_Maguire_20180517_ALL/ReadMe_SDNET2018 .txt`.
- User-provided project report: `project_report.txt`.
- [Ultralytics classification and custom transforms](https://docs.ultralytics.com/tasks/classify/).
- [PyTorch installation](https://pytorch.org/get-started/locally/).


"""Checks for leakage and spatial coverage that matter to this inspection pipeline."""
import csv
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import crack_vision


class DatasetChecks(unittest.TestCase):
    def test_source_groups_and_exact_images_never_cross_splits(self):
        with (ROOT / "dataset/manifest.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for key in ("group", "sha256"):
            assignments = {}
            for row in rows:
                assignments.setdefault(row[key], set()).add(row["split"])
            self.assertTrue(all(len(value) == 1 for value in assignments.values()), key)
        for split in ("train", "val", "test"):
            self.assertEqual({row["label"] for row in rows if row["split"] == split}, {"crack", "no_crack"})

    def test_conflicting_duplicate_is_absent(self):
        audit = json.loads((ROOT / "dataset/duplicate_audit.json").read_text())
        with (ROOT / "dataset/manifest.csv").open(encoding="utf-8") as handle:
            included = {row["sha256"] for row in csv.DictReader(handle)}
        for entry in audit:
            if "exclude_all_copies" in entry["reason"]:
                self.assertNotIn(entry["first"]["sha256"], included)


class LocalizationChecks(unittest.TestCase):
    def test_inference_tensor_matches_training_validation_preprocessing(self):
        import cv2
        import torch
        from ultralytics.cfg import get_cfg
        from crack_training import FullPatchDataset
        args = get_cfg(overrides={"imgsz": 224, "cache": False})
        dataset = FullPatchDataset(ROOT / "dataset/val", args, augment=False, prefix="val")
        expected = dataset[0]["img"]
        frame = cv2.imread(dataset.samples[0][0])
        class SpyModel:
            names = {0: "crack", 1: "no_crack"}
            def predict(self, tensors, **kwargs):
                self.tensors = tensors
                return [SimpleNamespace(probs=SimpleNamespace(data=torch.tensor([0.7, 0.3])))]
        model = SpyModel()
        crack_vision.scores(model, [frame], "cpu")
        torch.testing.assert_close(model.tensors[0], expected, rtol=0, atol=0)

    def test_every_pixel_including_frame_edges_is_covered(self):
        import numpy as np
        frame = np.zeros((301, 517, 3), dtype=np.uint8)
        args = SimpleNamespace(patch=256, stride=128, min_area=1, device="cpu", batch=24)
        with patch.object(crack_vision, "scores", side_effect=lambda model, images, device, batch: np.full(len(images), 0.8)):
            _, heat, metadata = crack_vision.candidate_regions(frame, None, args, 0.6)
        np.testing.assert_allclose(heat, 0.8, atol=1e-6)
        self.assertEqual(len(metadata["boxes"]), 1)
        self.assertEqual(metadata["boxes"][0]["width"], 517)
        self.assertEqual(metadata["boxes"][0]["height"], 301)

    def test_small_negative_frame_has_no_candidates(self):
        import numpy as np
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        args = SimpleNamespace(patch=256, stride=128, min_area=1, device="cpu", batch=24)
        with patch.object(crack_vision, "scores", side_effect=lambda model, images, device, batch: np.full(len(images), 0.1)):
            _, _, metadata = crack_vision.candidate_regions(frame, None, args, 0.5)
        self.assertEqual(metadata["boxes"], [])


if __name__ == "__main__":
    unittest.main()

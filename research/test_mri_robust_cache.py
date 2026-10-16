import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cache_mri_rgb import file_checks, pose_features
from cache_mri_robust_rgb import CONDITIONS, ROOT, expanded_ids, main, pixel_probes, validate_completed, verify_archive_directory


class RobustPixelCacheChecks(unittest.TestCase):
    def test_expansion_preserves_old_actual_ids_without_duplicates(self):
        samples = np.arange(256) * 13 + 7
        old = samples[np.linspace(0, 255, 64, dtype=int)]
        ids, reused = expanded_ids(samples, old)
        self.assertEqual(len(ids), 96)
        self.assertEqual(int(reused.sum()), 64)
        np.testing.assert_array_equal(ids[reused], old)
        self.assertEqual(len(np.unique(ids)), 96)
        self.assertTrue(np.isin(ids, samples).all())
        self.assertTrue(np.all(np.diff(ids) > 0))

    def test_48_frame_source_cannot_claim_64_reused_frames(self):
        samples = np.arange(256)
        for old in (samples[:48], samples[:64], np.zeros(64, dtype=int)):
            with self.assertRaises(ValueError):
                expanded_ids(samples, old)

    def test_noise_replays_by_frame_and_changes_with_subject_or_frame(self):
        image = np.full((120, 160, 3), 128, dtype=np.uint8)
        first, seed = pixel_probes(image, 1, 10)
        replay, replay_seed = pixel_probes(image, 1, 10)
        np.testing.assert_array_equal(first["noisy_dark"], replay["noisy_dark"])
        self.assertEqual(seed, replay_seed)
        self.assertEqual(seed, [20261013, 1, 10])
        for subject, frame in ((2, 10), (1, 11)):
            changed, _ = pixel_probes(image, subject, frame)
            self.assertFalse(np.array_equal(first["noisy_dark"], changed["noisy_dark"]))
        self.assertEqual(first["noisy_dark"].dtype, np.uint8)
        self.assertGreater(float(first["noisy_dark"].std()), 7.)
        self.assertLess(float(first["noisy_dark"].mean()), 20.)
        np.testing.assert_array_equal(image, np.full_like(image, 128))

    def test_occlusion_uses_same_pixel_geometry_without_gt_or_seed(self):
        image = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
        first, _ = pixel_probes(image, 1, 10)
        other, _ = pixel_probes(image, 20, 999)
        np.testing.assert_array_equal(first["occlusion"], other["occlusion"])
        np.testing.assert_array_equal(first["occlusion"][15:90, 80:120], np.full((75, 40, 3), 114))
        outside = np.ones((100, 200), dtype=bool)
        outside[15:90, 80:120] = False
        np.testing.assert_array_equal(first["occlusion"][outside], image[outside])

    def test_detector_missingness_retains_every_selected_frame(self):
        poses = np.zeros((96, 17, 3), dtype=np.float32)
        features, available = pose_features(poses, [1000, 1000])
        self.assertEqual(features.shape, (96, 51))
        self.assertEqual(available.shape, (96, 17))
        self.assertEqual(int((~available[:, 11]).sum()), 96)

    def test_archive_directory_crc_is_checked_without_reading_payload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "video.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("subject1_color0.mp4", b"authenticated-video")
            with zipfile.ZipFile(path) as archive:
                info = archive.infolist()[0]
                entry = {"member": info.filename, "bytes": info.file_size,
                         "crc32": info.CRC, "compression_method": info.compress_type}
            with patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("Payload must not be read")):
                verify_archive_directory(path, [entry], path.stat().st_size)
            for changed in ({**entry, "crc32": entry["crc32"] ^ 1}, {**entry, "bytes": entry["bytes"] + 1}):
                with self.assertRaises(ValueError):
                    verify_archive_directory(path, [changed], path.stat().st_size)
            with self.assertRaises(ValueError):
                verify_archive_directory(path, [entry], path.stat().st_size + 1)

    def test_completed_subject_resume_rejects_changed_provenance_or_content(self):
        ids = np.arange(96)
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "subject1.npz"
            values = {"frame_ids": ids, "protocol_sha256": "fixed-protocol", "old_cache_sha256": "old-cache",
                      "source_sha256": "source-video"}
            for condition in CONDITIONS:
                values[condition + "_poses"] = np.zeros((96, 17, 3))
                values[condition + "_features"] = np.zeros((96, 51))
                values[condition + "_available"] = np.zeros((96, 17), dtype=bool)
                values[condition + "_missing"] = np.ones(96, dtype=bool)
            np.savez(path, **values)
            row = {"cache_sha256": file_checks(path)[0]}
            validate_completed(path, row, ids, "fixed-protocol", "old-cache", "source-video")
            for expected in ((ids, "changed-protocol", "old-cache", "source-video"),
                             (ids, "fixed-protocol", "changed-old", "source-video"),
                             (ids, "fixed-protocol", "old-cache", "changed-video"),
                             (ids + 1, "fixed-protocol", "old-cache", "source-video")):
                with self.assertRaises(ValueError):
                    validate_completed(path, row, *expected)
            with path.open("ab") as handle:
                handle.write(b"changed-bytes")
            with self.assertRaises(ValueError):
                validate_completed(path, row, ids, "fixed-protocol", "old-cache", "source-video")

    def test_source_cache_cannot_be_used_as_output(self):
        source = ROOT / "artifacts/research_execution_20261012/rgb_features"
        with patch("sys.argv", ["cache_mri_robust_rgb.py", "--output", str(source)]):
            with self.assertRaisesRegex(ValueError, "separate"):
                main()


if __name__ == "__main__":
    unittest.main()

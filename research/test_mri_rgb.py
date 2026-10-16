import io
import unittest
import zipfile

import numpy as np

from cache_mri_rgb import independent_pose, pose_features, projection_errors
from run_mri_baselines import radar_statistics
from run_mri_rgb_baselines import shifted_radar


class RGBReaderChecks(unittest.TestCase):
    def test_injected_lag_uses_timestamps_without_wrapping_record_edges(self):
        features = np.zeros((13, 14, 14, 5))
        features[:, 0, 0, 0] = np.arange(13) + 1
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for member, array in (("raw_data/unixtime/subject1_unix.npy", np.arange(13) * 100),
                                  ("features/radar/subject1_featuremap.npy", features)):
                payload = io.BytesIO()
                np.save(payload, array, allow_pickle=False)
                archive.writestr("dataset_release/" + member, payload.getvalue())
        labels = {"radar_avail_frames": [0, 12], "video_label": {"motion": [0, 13]}}
        with zipfile.ZipFile(buffer) as archive:
            shifted, common = shifted_radar(archive, labels, 1, np.array([0, 5, 12]))
        np.testing.assert_array_equal(common, [False, True, False])
        self.assertEqual(shifted[-500]["statistics"][1, 0], 1.)
        self.assertEqual(shifted[0]["statistics"][1, 0], 6.)
        self.assertEqual(shifted[500]["statistics"][1, 0], 11.)

    def test_radar_padding_and_structural_missingness(self):
        points = np.zeros((2, 196, 5))
        points[0, 0] = [1, 2, 3, 4, 5]
        points[0, 1] = [3, 4, 5, 6, 7]
        statistics, observed = radar_statistics(points)
        np.testing.assert_array_equal(observed, [True, False])
        np.testing.assert_array_equal(statistics[0, :5], [2, 3, 4, 5, 6])
        np.testing.assert_array_equal(statistics[0, 5:10], np.ones(5))
        np.testing.assert_array_equal(statistics[1], np.zeros(21))
        self.assertAlmostEqual(statistics[0, -1], 2 / 196)

    def test_missing_roots_cannot_create_hip_relative_coordinates(self):
        poses = np.ones((2, 17, 3), dtype=float)
        poses[0, :, 0] = np.arange(17)
        poses[1, 11, 2] = 0
        features, available = pose_features(poses, [100, 200])
        self.assertTrue(available[0].all())
        self.assertFalse(available[1].any())
        np.testing.assert_array_equal(features[1], np.zeros(51))
        self.assertAlmostEqual(features[0, 0], -11.5 / 100)

    def test_invalid_input_is_rejected(self):
        for poses, focal in ((np.zeros((2, 33, 3)), [1, 1]),
                             (np.full((2, 17, 3), np.nan), [1, 1]),
                             (np.zeros((2, 17, 3)), [0, 1])):
            with self.assertRaises(ValueError):
                pose_features(poses, focal)

    def test_full_frame_bgr_preprocessing_and_missing_detection(self):
        class FakeModel:
            def get_inputs(self):
                return [type("Input", (), {"name": "input"})()]

            def run(self, _, inputs):
                data = inputs["input"]
                np.testing.assert_array_equal(data[0, :, 0, 0], [10, 20, 30])
                np.testing.assert_array_equal(data[0, :, -1, -1], [114, 114, 114])
                self.last = data
                return np.zeros((1, 0, 5)), np.zeros((1, 0, 17, 3))

        image = np.tile(np.array([10, 20, 30], dtype=np.uint8), (100, 200, 1))
        np.testing.assert_array_equal(independent_pose(FakeModel(), image), np.zeros((17, 3)))

    def test_calibration_projection_units_and_axis_direction(self):
        camera_points = np.tile(np.array([2., 3., 10.]), (1, 17, 1))
        stored = camera_points * [-.001, -.001, .001]
        projection = np.column_stack([np.eye(3), np.zeros(3)])
        labels = {"refined_gt_kps": stored.transpose(0, 2, 1),
                  "pose_2d_l": np.tile(np.array([.2, .3]), (1, 17, 1)).transpose(0, 2, 1),
                  "pose_2d_r": np.tile(np.array([.2, .3]), (1, 17, 1)).transpose(0, 2, 1),
                  "camera_matrix": {"cam1_proj_mat": projection, "cam2_proj_mat": projection}}
        for result in projection_errors(labels, [0]).values():
            self.assertAlmostEqual(result["median_px"], 0.)


if __name__ == "__main__":
    unittest.main()

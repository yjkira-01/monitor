"""The radar-only control must receive identical inputs across pose encodings."""

import unittest

from run_fusion_experiments_gpu import augmented_batch
import torch


class RepresentationRadarScheduleChecks(unittest.TestCase):
    def test_shared_hip_reference_keeps_radar_identical_when_visibility_changes(self):
        count = 256
        source_rng = torch.Generator().manual_seed(20261013)
        points = torch.randn(count, 196, 5, generator=source_rng)
        mask = torch.ones(count, 196, dtype=torch.bool)
        hip, centroid, references = [], [], []
        for step in (3, 4):
            hip_confidence = torch.ones(count, 17) * .8
            hip_confidence[::step] = 0
            centroid_confidence = torch.ones(count, 17) * .8
            hip.append((torch.zeros(count, 51), points, mask, hip_confidence))
            centroid.append((torch.ones(count, 51), points, mask, centroid_confidence))
            references.append((hip_confidence >= .3).any(1))
        reference = torch.stack(references)
        indices = torch.randperm(count, generator=source_rng)
        hip_batch = augmented_batch(hip, indices, torch.Generator().manual_seed(17), reference)
        centroid_batch = augmented_batch(centroid, indices, torch.Generator().manual_seed(17), reference)
        torch.testing.assert_close(hip_batch[1], centroid_batch[1], atol=0, rtol=0)
        torch.testing.assert_close(hip_batch[2], centroid_batch[2], atol=0, rtol=0)
        # Establish that this case actually exposes the original confound.
        uncontrolled_hip = augmented_batch(hip, indices, torch.Generator().manual_seed(17))
        uncontrolled_centroid = augmented_batch(centroid, indices, torch.Generator().manual_seed(17))
        self.assertTrue((uncontrolled_hip[2] != uncontrolled_centroid[2]).any())
        self.assertTrue((hip_batch[3] != centroid_batch[3]).any())


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from PIL import Image

from edit_geometry import (
    centered_grid_offsets,
    prepare_reference_image,
    reference_transform,
    validate_edit_ref_geometry,
)
from train_t2i import build_pos_mask


class EditGeometryTests(unittest.TestCase):
    def test_default_square_stretches_to_target(self):
        image = Image.new("RGB", (1600, 900))
        out = prepare_reference_image(image, (768, 768), geometry="square")
        self.assertEqual(out.size, (768, 768))

    def test_fit_wide_reference_without_squash(self):
        transform = reference_transform((1600, 900), (768, 768), align=16)
        self.assertEqual(transform.size, (768, 432))
        crop_w = transform.crop[2] - transform.crop[0]
        crop_h = transform.crop[3] - transform.crop[1]
        self.assertAlmostEqual(crop_w / crop_h, 768 / 432, places=2)

    def test_near_matching_ratio_fills_target(self):
        transform = reference_transform((1000, 950), (768, 768), align=16)
        self.assertEqual(transform.size, (768, 768))
        self.assertLess(transform.crop[2] - transform.crop[0], 1000)

    def test_fractional_center_is_preserved(self):
        self.assertEqual(
            centered_grid_offsets((48, 48), [(48, 27), (31, 48)]),
            [(0.0, 10.5), (8.5, 0.0)],
        )

    def test_position_builder_applies_fractional_offsets_only_to_refs(self):
        pos, _ = build_pos_mask(
            4, 4, torch.ones(1, 1, dtype=torch.bool),
            ref_grids=[(4, 3)], center_refs=True)
        ref = pos[0, 1:13]
        target = pos[0, 13:]
        self.assertTrue(torch.all(ref[:, 2].frac() == 0.5))
        self.assertTrue(torch.all(target[:, 1:].frac() == 0.0))

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            validate_edit_ref_geometry("stretch-ish")


if __name__ == "__main__":
    unittest.main()

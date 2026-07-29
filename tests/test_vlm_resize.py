import random
import unittest

from PIL import Image

from sample_edit import _vlm_resize


class VlmResizeTests(unittest.TestCase):
    def test_fixed_cap_preserves_aspect(self):
        out = _vlm_resize(Image.new("RGB", (1600, 800)), 400)
        self.assertEqual(out.size, (400, 200))

    def test_seeded_jitter_is_bounded_and_reproducible(self):
        a = _vlm_resize(Image.new("RGB", (1600, 800)), 768, 384, random.Random(7))
        b = _vlm_resize(Image.new("RGB", (1600, 800)), 768, 384, random.Random(7))
        self.assertEqual(a.size, b.size)
        self.assertGreaterEqual(max(a.size), 384)
        self.assertLessEqual(max(a.size), 768)

    def test_small_images_are_not_upscaled(self):
        out = _vlm_resize(Image.new("RGB", (320, 200)), 768, 384, random.Random(2))
        self.assertEqual(out.size, (320, 200))


if __name__ == "__main__":
    unittest.main()

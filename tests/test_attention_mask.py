import unittest

import torch
import torch.nn.functional as F

from mmdit import _mask


class CompactAttentionMaskTests(unittest.TestCase):
    def test_key_only_mask_matches_dense_mask_on_retained_queries(self):
        torch.manual_seed(0)
        q = torch.randn(2, 3, 7, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        valid = torch.tensor([
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False],
        ])
        dense = valid[:, None, :, None] & valid[:, None, None, :]
        compact = _mask(valid)

        dense_out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense)
        compact_out = F.scaled_dot_product_attention(q, k, v, attn_mask=compact)

        self.assertEqual(compact.shape, (2, 1, 1, 7))
        for batch, length in enumerate(valid.sum(dim=1).tolist()):
            torch.testing.assert_close(
                compact_out[batch, :, :length], dense_out[batch, :, :length])


if __name__ == "__main__":
    unittest.main()

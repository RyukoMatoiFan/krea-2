import unittest
from unittest.mock import patch

import torch
from torch import nn

from mmdit import SingleStreamDiT


class _BlockContainer:
    compile_blocks = SingleStreamDiT.compile_blocks

    def __init__(self):
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self._swap_blocks = set()


class CompileBlocksTests(unittest.TestCase):
    def test_each_block_is_compiled_with_requested_mode(self):
        model = _BlockContainer()

        def wrap(module, mode):
            wrapped = nn.Sequential(module)
            wrapped.compile_mode = mode
            return wrapped

        with patch.object(torch, "compile", side_effect=wrap) as compiler:
            count = model.compile_blocks("reduce-overhead")
        self.assertEqual(count, 2)
        self.assertEqual(compiler.call_count, 2)
        self.assertEqual(model.blocks[0].compile_mode, "reduce-overhead")

    def test_block_swap_combination_is_rejected(self):
        model = _BlockContainer()
        model._swap_blocks = {1}
        with self.assertRaises(RuntimeError):
            model.compile_blocks()


if __name__ == "__main__":
    unittest.main()

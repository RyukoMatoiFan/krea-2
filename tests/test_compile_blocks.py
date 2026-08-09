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
        originals = list(model.blocks)

        def wrap(fn, mode, fullgraph, dynamic):
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)
            wrapped.compile_mode = mode
            wrapped.fullgraph = fullgraph
            wrapped.dynamic = dynamic
            return wrapped

        with patch.object(torch, "compile", side_effect=wrap) as compiler:
            count = model.compile_blocks("reduce-overhead", dynamic=True)
        self.assertEqual(count, 2)
        self.assertEqual(compiler.call_count, 2)
        self.assertIs(model.blocks[0], originals[0])
        self.assertEqual(model.blocks[0]._compiled_forward.compile_mode, "reduce-overhead")
        self.assertTrue(model.blocks[0]._compiled_forward.fullgraph)
        self.assertTrue(model.blocks[0]._compiled_forward.dynamic)
        self.assertNotIn("_orig_mod", " ".join(model.blocks.state_dict()))

    def test_block_swap_combination_is_rejected(self):
        model = _BlockContainer()
        model._swap_blocks = {1}
        with self.assertRaises(RuntimeError):
            model.compile_blocks()


if __name__ == "__main__":
    unittest.main()

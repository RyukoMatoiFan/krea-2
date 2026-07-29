import os
import tempfile
import unittest

import torch
from torch import nn

from lora import LoRALinear, load_lora_stage_init, save_lora


class LoraStageInitTests(unittest.TestCase):
    def test_weights_only_stage_init_loads_adapter(self):
        source = {"layer": LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)}
        with torch.no_grad():
            source["layer"].lora_B.fill_(0.25)
        target = {"layer": LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stage.safetensors")
            save_lora(source, path, rank=2, alpha=2)
            load_lora_stage_init(target, {}, path)
        self.assertTrue(torch.allclose(
            target["layer"].lora_B, source["layer"].lora_B.to(torch.bfloat16).float(),
            atol=2e-3))

    def test_missing_file_is_rejected(self):
        target = {"layer": LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)}
        with self.assertRaises(SystemExit):
            load_lora_stage_init(target, {}, "missing-stage.safetensors")


if __name__ == "__main__":
    unittest.main()

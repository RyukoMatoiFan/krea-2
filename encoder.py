import contextlib
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)


@dataclass
class TextEncoderConfig:
    model_id: str
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ):
        super().__init__()
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(version)
        self._version = version
        self._mm_processor = None  # lazily built multimodal processor (image conditioning)
        self.tokenizer = AutoTokenizer.from_pretrained(version, max_length=max_length)
        self.processor = Qwen2TokenizerFast.from_pretrained(
            version, max_length=max_length
        )
        self.qwen = self.qwen.eval().requires_grad_(False)
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    # System prompt content shared by the text and image-conditioning paths.
    SYSTEM_PROMPT = ("Describe the image by detailing the color, shape, size, texture, quantity, "
                     "text, spatial relationships of the objects and background:")
    VISION_START_TOKEN_ID = 151652  # <|vision_start|> (Qwen3-VL)

    def forward(self, text, images=None):
        if images is not None:                       # condition on reference image(s)
            return self._forward_mm(text, images)
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(
            self.qwen.device, non_blocking=True
        )
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        # Gradients reach the encoder only when it is trainable (joint TE fine-tune, te_lr>0); the
        # default frozen-inference path stays under no_grad.
        trainable = any(p.requires_grad for p in self.qwen.parameters())
        with contextlib.nullcontext() if trainable else torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length
                + prefix_idx
                - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(
                input_ids=input_ids, attention_mask=mask, output_hidden_states=True
            )

            hiddens = torch.stack(
                [states.hidden_states[i] for i in self.select_layers], dim=2
            )
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

    def _forward_mm(self, text, images) -> tuple[Tensor, Tensor]:
        """Condition on reference image(s) via the Qwen3-VL vision tower.

        Each prompt + its reference image go through the multimodal processor (image-pad tokens are
        expanded by the vision grid); we tap the same 12 layers and keep the image + user-text tokens,
        dropping only the system prefix -> conditioning lands in the same ``(B, L, 12, 2560)`` space the
        DiT consumes. ``images`` is one PIL image per prompt (or a single shared image). The DiT must be
        (LoRA-)trained to use these image-conditioning tokens, which the text-only base does not produce.
        Assumes one image per prompt (batch=1 is the typical style/edit case).
        """
        from transformers import AutoProcessor

        if self._mm_processor is None:
            self._mm_processor = AutoProcessor.from_pretrained(self._version)
        if not isinstance(images, (list, tuple)):
            images = [images] * len(text)
        chats = [
            self._mm_processor.apply_chat_template(
                [{"role": "system", "content": self.SYSTEM_PROMPT},
                 {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                tokenize=False, add_generation_prompt=True)
            for t in text
        ]
        inputs = self._mm_processor(text=chats, images=list(images), return_tensors="pt",
                                    padding=True).to(self.qwen.device)
        trainable = any(p.requires_grad for p in self.qwen.parameters())
        with contextlib.nullcontext() if trainable else torch.no_grad():
            states = self.qwen(**inputs, output_hidden_states=True)
            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
        mask = inputs["attention_mask"].bool()
        row = inputs["input_ids"][0]
        vs = (row == self.VISION_START_TOKEN_ID).nonzero()
        start = int(vs[0]) if len(vs) else 0       # keep [vision_start .. user text]; drop system prefix
        return hiddens[:, start:], mask[:, start:]

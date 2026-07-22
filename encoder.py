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
        system_prompt: str = "",
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
        # Default (system_prompt="") reproduces Krea 2's native descriptor BYTE-FOR-BYTE and keeps the
        # validated drop index 34. An edit-framed prompt (e.g. "Edit the reference image according to
        # the instruction:") has a different token count, so derive the drop index from the tokenizer.
        sysp = system_prompt or self.SYSTEM_PROMPT
        self.system_prompt = sysp
        self.prompt_template_encode_prefix = f"<|im_start|>system\n{sysp}<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = (
            34 if not system_prompt
            else len(self.tokenizer(self.prompt_template_encode_prefix)["input_ids"])
        )
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

    def _patch_vision_patch_embed(self) -> int:
        """The vision tower's patch_embed is a Conv3d whose kernel == stride, i.e. a plain linear
        projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and falls back to a
        slow path, so route it through the equivalent F.linear (a GEMM). The weight is read lazily so
        the patch survives later .to(device/dtype) moves. Returns the number of modules patched."""
        patched = 0
        for module in self.qwen.modules():
            proj = getattr(module, "proj", None)
            if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(proj.stride):
                def fast_forward(hidden_states, _proj=proj):
                    w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                    x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                    return torch.nn.functional.linear(x, w, _proj.bias)
                module.forward = fast_forward
                patched += 1
        return patched

    def _forward_mm(self, text, images) -> tuple[Tensor, Tensor]:
        """Condition on reference image(s) via the Qwen3-VL vision tower.

        Each prompt + its reference image(s) go through the multimodal processor (image-pad tokens are
        expanded by the vision grid); we tap the same 12 layers and keep the image + user-text tokens,
        dropping only the system prefix -> conditioning lands in the same ``(B, L, 12, 2560)`` space the
        DiT consumes. The DiT must be (LoRA-)trained to use these image-conditioning tokens, which the
        text-only base does not produce.

        ``images`` accepts three shapes:

        * a single PIL image                -> shared by every prompt
        * ``[img, img, ...]``               -> ONE image per prompt (single-reference editing)
        * ``[[img, img], [img], ...]``      -> N images per prompt (multi-reference composition)

        The nested form is what makes instructions like "put the woman from image_2 onto the
        background of image_1" groundable: with only the first reference visible the encoder cannot
        resolve which object the text is referring to. Placeholders are emitted in reference order,
        so image_1/image_2 in the caption line up with refs[0]/refs[1].
        """
        from transformers import AutoProcessor

        if self._mm_processor is None:
            self._mm_processor = AutoProcessor.from_pretrained(self._version)
            self._patch_vision_patch_embed()
        # Normalise every input shape to one LIST of images per prompt.
        if not isinstance(images, (list, tuple)):
            per_prompt = [[images] for _ in text]
        elif images and isinstance(images[0], (list, tuple)):
            per_prompt = [list(g) for g in images]
        else:
            per_prompt = [[im] for im in images]
        if len(per_prompt) == 1 and len(text) > 1:      # single shared group -> broadcast
            per_prompt = [list(per_prompt[0]) for _ in text]
        if len(per_prompt) != len(text):
            raise ValueError(f"images groups ({len(per_prompt)}) != prompts ({len(text)})")
        chats = [
            self._mm_processor.apply_chat_template(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": [*({"type": "image"} for _ in imgs),
                                              {"type": "text", "text": t}]}],
                tokenize=False, add_generation_prompt=True)
            for t, imgs in zip(text, per_prompt)
        ]
        # Flat, in prompt-then-reference order: the processor consumes images positionally against
        # the image placeholders it finds across the batch.
        flat = [im for group in per_prompt for im in group]
        inputs = self._mm_processor(text=chats, images=flat, return_tensors="pt",
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

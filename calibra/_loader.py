"""Internal HuggingFace model loading utilities."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def _load_model(
    model_name: str,
    device: str | None = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> tuple:
    """Load a HuggingFace causal language model and its tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"Qwen/Qwen3-8B"``).
        device: Target device. Defaults to ``"cuda"`` when available, else ``"cpu"``.
        load_in_8bit: Enable 8-bit quantization (requires CUDA and bitsandbytes).
        load_in_4bit: Enable 4-bit quantization (requires CUDA and bitsandbytes).

    Returns:
        A ``(model, tokenizer)`` tuple with the model in eval mode.

    Raises:
        ValueError: If both ``load_in_8bit`` and ``load_in_4bit`` are True.
    """
    if load_in_8bit and load_in_4bit:
        raise ValueError("Specify at most one of load_in_8bit or load_in_4bit.")

    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "trust_remote_code": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    quantization_active = False
    if (load_in_8bit or load_in_4bit) and device != "cuda":
        logger.warning("Quantization requires CUDA; ignoring quantization flags on CPU.")
    elif load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        quantization_active = True
    elif load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        quantization_active = True

    if not quantization_active:
        model_kwargs["torch_dtype"] = torch_dtype

    if _flash_attn_available(device):
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("FlashAttention 2 enabled.")

    logger.info("Loading model: %s (device=%s)", model_name, device)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device == "cpu":
        model = model.to(device)

    model.eval()
    logger.info("Model ready.")
    return model, tokenizer


def _flash_attn_available(device: str) -> bool:
    if device != "cuda" or not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

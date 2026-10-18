"""Reload the saved pilot adapter and compare a fixed validation prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paired-evaluation", action="store_true")
    args = parser.parse_args()
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from qwen_finetune import MODEL_ID, MODEL_REVISION, encode, evaluate, load_rows, save, sha
    from safetensors.torch import load_file
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    adapter = args.run / "adapter"
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    b_weights = [value for key, value in tensors.items() if ".lora_B." in key]
    if not b_weights or not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Missing or non-finite adapter weights")
    if not any(torch.count_nonzero(value).item() for value in b_weights):
        raise ValueError("Adapter B weights have not changed from zero initialization")
    processor = AutoProcessor.from_pretrained(adapter)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"],
        ),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    row = load_rows(args.data, "validation")[0]
    inputs = encode(processor, row, args.data, answer=False).to("cuda")
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    output = processor.decode(
        result[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    expected = json.loads(
        (args.run / "after_windows.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    verification = {
        "status": "PASS" if output == expected["output"] else "MISMATCH",
        "scope": "One fixed validation window; serialization smoke, not an accuracy score",
        "window_id": row["id"],
        "same_prediction_after_reload": output == expected["output"],
        "adapter_weights_finite": True,
        "lora_B_weights_changed": True,
        "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
        "execution_source_sha256": sha(Path(__file__).with_name("qwen_finetune.py")),
    }
    save(args.run / "reload_verification.json", verification)
    print(json.dumps(verification), flush=True)
    if args.paired_evaluation:
        rows = load_rows(args.data, "validation")
        # Same loaded model, frozen weights, precision, frames and prompts;
        # only the trained LoRA adapter is toggled for this paired comparison.
        with model.disable_adapter():
            baseline = evaluate(model, processor, args.data, rows, args.run, "paired_base")
        adapted = evaluate(model, processor, args.data, rows, args.run, "paired_adapter")
        save(
            args.run / "paired_comparison.json",
            {
                "status": "FINISHED",
                "same_loaded_model_and_precision": True,
                "base": baseline["metrics"],
                "adapter": adapted["metrics"],
                "base_wall_sec": baseline["wall_sec"],
                "adapter_wall_sec": adapted["wall_sec"],
                "validation_sha256": sha(args.data / "validation.jsonl"),
                "scope": "Internal validation8; only LoRA enable/disable differs",
                "note": (
                    "Original before pass preceded PEFT dtype preparation; use this comparison."
                ),
            },
        )


if __name__ == "__main__":
    main()

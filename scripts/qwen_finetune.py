"""Standalone cloud QLoRA binary pilot; no uploads, telemetry or API deployment.

Run from an extracted private bundle. Validation labels are clip-level and are
never used for training loss or annotation-dependent frame/window selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT = (
    "Кадры из одного фрагмента камеры банковской кассы идут по порядку времени. "
    "Определи, видно ли нарушение правил кассового обслуживания. "
    "Например: деньги предыдущего клиента остались открытыми при обслуживании следующего; "
    "кассир оставил рабочее место во время обслуживания клиента. "
    "Учитывай последовательность действий, присутствие клиента и обращение с деньгами. "
    "Ответь только одной цифрой: 1 — нарушение видно, 0 — нарушение не видно."
)


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(root: Path, split: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Empty dataset")
    for row in rows:
        if row["label"] not in (0, 1) or not 1 <= len(row["images"]) <= 16:
            raise ValueError("Invalid sample")
        for name in row["images"]:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Missing or out-of-bundle image")
    return rows


def verify_bundle(root: Path, train: list, validation: list) -> None:
    train_keys = {r["video_key"] for r in train}
    validation_keys = {r["video_key"] for r in validation}
    if train_keys & validation_keys:
        raise ValueError("Train/validation source overlap")
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    hashes = [r["source_sha256"] for r in provenance]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate source groups")
    if {r["key"] for r in provenance} != train_keys | validation_keys:
        raise ValueError("Unexpected video group in bundle")
    for record in provenance:
        for frame in record["frames"]:
            if sha(root / "frames" / record["key"] / frame["file"]) != frame["sha256"]:
                raise ValueError("Frame checksum mismatch")


def user_message(count: int) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in range(count)],
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def encode(processor, row: dict, root: Path, *, answer: bool):
    from PIL import Image

    images = []
    for name in row["images"]:
        with Image.open(root / name) as opened:
            images.append(opened.convert("RGB"))
    messages = user_message(len(images))
    if answer:
        messages += [
            {"role": "assistant", "content": [{"type": "text", "text": str(row["label"])}]}
        ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=not answer)
    encoded = processor(text=[text], images=images, return_tensors="pt", padding=False)
    if encoded["input_ids"].shape[1] > 10000:
        raise ValueError("Visual/text token budget exceeded; refusing silent truncation")
    return encoded


class BinaryCollator:
    def __init__(self, processor, root: Path):
        self.processor = processor
        self.root = root

    def __call__(self, rows):
        # Batch one avoids padding/multimodal position mixing in this initial pilot.
        # Gradient accumulation supplies the effective batch size.
        import torch

        if len(rows) != 1:
            raise ValueError("Pilot uses microbatch one")
        row = rows[0]
        encoded = encode(self.processor, row, self.root, answer=True)
        prefix = encode(self.processor, row, self.root, answer=False)["input_ids"]
        length = prefix.shape[1]
        if not torch.equal(encoded["input_ids"][:, :length], prefix):
            raise ValueError("Chat template prefix mismatch; refusing incorrect loss mask")
        labels = encoded["input_ids"].clone()
        labels[:, :length] = -100
        if not (labels != -100).any():
            raise ValueError("No supervised answer tokens")
        encoded["labels"] = labels
        return encoded


def metrics(rows: list[dict]) -> dict:
    complete = [r for r in rows if r["prediction"] is not None]
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in complete)
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in complete)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in complete)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in complete)
    return {
        "videos": len(rows),
        "complete": len(complete),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "coverage": len(complete) / len(rows),
        "accuracy_complete_only": (tp + tn) / len(complete) if complete else None,
        "precision_complete_only": tp / (tp + fp) if tp + fp else None,
        "recall_complete_only": tp / (tp + fn) if tp + fn else None,
        "f1_complete_only": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "correct_per_attempted": (tp + tn) / len(rows),
    }


def evaluate(model, processor, root: Path, rows: list, output: Path, stage: str) -> dict:
    import torch

    model.eval()
    groups = defaultdict(list)
    started = time.monotonic()
    with (output / f"{stage}_windows.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            # label and annotation metadata are intentionally absent from encode(answer=False).
            inputs = encode(processor, row, root, answer=False).to("cuda")
            with torch.inference_mode():
                result = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            text = processor.decode(
                result[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            prediction = int(text) if text in ("0", "1") else None
            groups[row["video_key"]].append((row["label"], prediction))
            stream.write(
                json.dumps(
                    {"id": row["id"], "prediction": prediction, "output": text}, ensure_ascii=False
                )
                + "\n"
            )
            stream.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {"stage": stage, "windows_done": index + 1, "windows_total": len(rows)}
                    ),
                    flush=True,
                )
    videos = []
    for key, values in groups.items():
        if len({label for label, _ in values}) != 1:
            raise ValueError("Conflicting video-level labels")
        predictions = [p for _, p in values]
        prediction = None if None in predictions else int(any(predictions))
        videos.append({"video_key": key, "label": values[0][0], "prediction": prediction})
    summary = {
        "stage": stage,
        "wall_sec": time.monotonic() - started,
        "metrics": metrics(videos),
        "video_results": videos,
        "selection": "Complete fixed sliding timeline; no annotation-time access",
        "independent_final_test": False,
    }
    save(output / f"{stage}_summary.json", summary)
    print(json.dumps({"stage": stage, "metrics": summary["metrics"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Only for explicitly named smoke"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    train = load_rows(args.data, "train")
    validation = load_rows(args.data, "validation")
    verify_bundle(args.data, train, validation)
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    set_seed(2026)
    random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, min_pixels=64 * 32 * 32, max_pixels=384 * 32 * 32
    )
    processor.tokenizer.padding_side = "right"
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
    configuration = {
        "base_model": MODEL_ID,
        "revision": MODEL_REVISION,
        "seed": 2026,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "microbatch": 1,
        "gradient_accumulation": args.accumulation,
        "learning_rate": 2e-5,
        "lora_r": 16,
        "lora_alpha": 32,
        "prompt": PROMPT,
        "train_sha256": sha(args.data / "train.jsonl"),
        "validation_sha256": sha(args.data / "validation.jsonl"),
        "provenance_sha256": sha(args.data / "provenance.json"),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]
        },
    }
    save(args.output / "configuration.json", configuration)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # PEFT may cast frozen non-quantized weights to float32. Establish the baseline
    # after that conversion so precision is identical before and after training.
    if not args.skip_baseline:
        evaluate(model, processor, args.data, validation, args.output, "before")
    model.config.use_cache = False
    targets = [
        name
        for name, _ in model.named_modules()
        if ".language_model." in name
        and name.rsplit(".", 1)[-1]
        in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ]
    if not targets:
        raise ValueError("No language-only LoRA targets found")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in name or ".language_model." not in name for name in trainable):
        raise ValueError("Unexpected trainable parameters")
    save(args.output / "trainable_parameters.json", trainable)

    class Progress(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            if logs:
                if "loss" in logs and not math.isfinite(logs["loss"]):
                    raise ValueError("Non-finite training loss")
                record = {
                    "step": state.global_step,
                    **logs,
                    "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                with (args.output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({"stage": "training", **record}), flush=True)

    arguments = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=2026,
        data_seed=2026,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        data_collator=BinaryCollator(processor, args.data),
        callbacks=[Progress()],
    )
    train_started = time.monotonic()
    result = trainer.train()
    training_sec = time.monotonic() - train_started
    model.save_pretrained(args.output / "adapter", safe_serialization=True)
    processor.save_pretrained(args.output / "adapter")
    save(
        args.output / "training_summary.json",
        {
            "status": "TRAINED",
            "global_steps": trainer.state.global_step,
            "training_sec": training_sec,
            "metrics": result.metrics,
            "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
    )
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    after = evaluate(model, processor, args.data, validation, args.output, "after")
    save(
        args.output / "run_summary.json",
        {
            "status": "FINISHED",
            "model_weights_trained": True,
            "adapter_saved": True,
            "total_wall_sec": time.monotonic() - started,
            "training_sec": training_sec,
            "after_metrics": after["metrics"],
            "api_deployed": False,
            "holdout_evaluated": False,
            "limitation": "Small binary pilot; event localization and explanations are not trained",
        },
    )


if __name__ == "__main__":
    main()

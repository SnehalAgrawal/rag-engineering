"""
Fine-tunes the chat model on ../data/training_qa.json with LoRA, and saves the adapter.

  python train.py

The base model is 4-bit, so this is QLoRA: the base weights stay frozen and
quantized, and only small low-rank adapter matrices are trained. The adapter is
written to ../adapters/, with the settings and the loss curve next to it.

Uses mlx-lm's trainer. The dataset class is our own only so that every example
is rendered with the same fixed date as evaluate.py (see DATE_STRING in common.py).
"""
import json
import random
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from common import ADAPTER_PATH, SYSTEM, TRAINING, model_path, prompt_tokens, training_answer

SETTINGS = {
    "seed": 0,
    "num_layers": 16,  # all 16 transformer layers of Llama 3.2 1B
    # mlx-lm's default LoRA settings. A first try with rank 16 at the same scale diverged:
    # the loss never went below its starting value.
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "batch_size": 4,
    # 120 examples / 4 per batch = 30 steps per pass. 100 steps (about 3.3 passes) was chosen from the
    # validation loss of a 300-step run: lowest at step 100 (1.103), then flat (1.15-1.21) to step 300.
    "iters": 100,
    "learning_rate": 1e-4,
    "max_seq_length": 512,
    "steps_per_eval": 25,
}


class ChatExamples:
    """Training examples as chat messages. Only the assistant's answer is trained on (the prompt is masked)."""

    def __init__(self, examples, tokenizer):
        self.examples = [
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": e["question"]},
             {"role": "assistant", "content": training_answer(e)}]
            for e in examples
        ]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def process(self, messages):
        tokens = prompt_tokens(self.tokenizer, messages, add_generation_prompt=False)
        offset = len(prompt_tokens(self.tokenizer, messages[:-1], add_generation_prompt=True))
        return tokens, offset


class LossLog(TrainingCallback):
    def __init__(self):
        self.train, self.valid = [], []

    def on_train_loss_report(self, info):
        self.train.append({"iteration": info["iteration"], "loss": round(float(info["train_loss"]), 4)})

    def on_val_loss_report(self, info):
        self.valid.append({"iteration": info["iteration"], "loss": round(float(info["val_loss"]), 4)})


def main() -> None:
    random.seed(SETTINGS["seed"])
    np.random.seed(SETTINGS["seed"])
    mx.random.seed(SETTINGS["seed"])

    model, tokenizer = load(model_path())
    train_examples = list(TRAINING["train"])
    random.shuffle(train_examples)  # batches are fixed groups of neighbors, so mix the articles first
    train_set = ChatExamples(train_examples, tokenizer)
    valid_set = ChatExamples(TRAINING["valid"], tokenizer)

    model.freeze()
    linear_to_lora_layers(model, SETTINGS["num_layers"], SETTINGS["lora_parameters"])
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    args = TrainingArgs(
        batch_size=SETTINGS["batch_size"], iters=SETTINGS["iters"], val_batches=-1,
        steps_per_report=10, steps_per_eval=SETTINGS["steps_per_eval"], steps_per_save=10**9,  # no intermediate checkpoints; the final adapter is always saved
        adapter_file=ADAPTER_PATH / "adapters.safetensors", max_seq_length=SETTINGS["max_seq_length"],
    )
    log = LossLog()
    start = time.perf_counter()
    train(model=model, optimizer=optim.Adam(learning_rate=SETTINGS["learning_rate"]), args=args,
          train_dataset=CacheDataset(train_set), val_dataset=CacheDataset(valid_set), training_callback=log)
    seconds = time.perf_counter() - start

    # mlx_lm.load(..., adapter_path=...) reads this file to rebuild the LoRA layers.
    (ADAPTER_PATH / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": SETTINGS["num_layers"],
        "lora_parameters": SETTINGS["lora_parameters"],
    }, indent=1))
    (ADAPTER_PATH / "training_log.json").write_text(json.dumps({
        "settings": SETTINGS, "train_examples": len(train_set), "valid_examples": len(valid_set),
        "trainable_parameters": int(trainable),
        "seconds": round(seconds, 1), "train_loss": log.train, "valid_loss": log.valid,
    }, indent=1))
    print(f"\nTrained {trainable:,} adapter parameters on {len(train_set)} examples in {seconds:.0f} s.")
    print(f"Validation loss: {log.valid[0]['loss']} before training, {log.valid[-1]['loss']} after.")
    print(f"Adapter saved to {ADAPTER_PATH}")


if __name__ == "__main__":
    main()

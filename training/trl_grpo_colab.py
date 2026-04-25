"""Colab-ready GRPO training script for EcoCloud War Room.

Run this on Colab / HF compute with:
    pip install trl transformers datasets accelerate peft bitsandbytes
    pip install -e .
    python training/trl_grpo_colab.py

This follows the Hugging Face TRL OpenEnv pattern:
https://huggingface.co/docs/trl/openenv
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from ecocloud_env.models import CloudAction
from ecocloud_env.server.environment import EcoCloudEnvironment

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/ecocloud-grpo"
TRAIN_PROMPTS = 256


def format_observation(env: EcoCloudEnvironment) -> str:
    """Render the current environment state as a compact text observation."""
    obs = env._state
    return (
        "EcoCloud state:\n"
        f"- latency_ms: {obs.latency:.1f}\n"
        f"- cost_usd: {obs.cost:.1f}\n"
        f"- carbon_units: {obs.carbon:.1f}\n"
        f"- load: {obs.load}\n"
        f"- step_count: {obs.step_count}\n"
        f"- stable_steps: {obs.stable_steps}\n"
        "Targets:\n"
        "- latency < 150\n"
        "- cost < 400\n"
        "- carbon < 220\n"
        "Choose the next cloud control action using the available tools."
    )


@dataclass
class EcoCloudToolEnv:
    """Expose EcoCloud as a multi-turn tool environment for GRPO."""

    # Track cumulative reward across the entire episode for richer signal
    cumulative_reward: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.env = EcoCloudEnvironment()
        self.reward = 0.0
        self.done = False
        self.cumulative_reward = 0.0

    def reset(self, **kwargs) -> str | None:
        """Start a fresh episode and return the initial observation."""
        seed = kwargs.get("seed")
        self.done = False
        self.reward = 0.0
        self.cumulative_reward = 0.0
        self.env.reset(seed=seed)
        return format_observation(self.env)

    def scale_up(self) -> str:
        """
        Add capacity to reduce latency quickly.
        Returns:
            Updated cloud observation after scaling up.
        """
        return self._apply("scale_up")

    def scale_down(self) -> str:
        """
        Remove capacity to reduce cost.
        Returns:
            Updated cloud observation after scaling down.
        """
        return self._apply("scale_down")

    def optimize_energy(self) -> str:
        """
        Optimize the current fleet for lower carbon and better efficiency.
        Returns:
            Updated cloud observation after local energy optimization.
        """
        return self._apply("optimize_energy")

    def migrate_region(self) -> str:
        """
        Shift the workload to a lower-carbon region.
        Returns:
            Updated cloud observation after migrating the workload.
        """
        return self._apply("migrate_region")

    def _apply(self, action: str) -> str:
        """Execute one environment action and surface the next observation."""
        if self.done:
            raise ValueError("Episode already finished.")
        result = self.env.step(CloudAction(action=action))
        self.reward = float(result.reward)
        self.cumulative_reward += self.reward
        self.done = bool(result.done)
        next_obs = result
        status = "SUCCESS" if next_obs.success else "IN_PROGRESS"
        if self.done:
            status = "EPISODE_COMPLETE"
        return (
            f"{status}\n"
            f"reward={self.reward:.1f}\n"
            f"cumulative_reward={self.cumulative_reward:.1f}\n"
            f"latency_ms={next_obs.latency:.1f}\n"
            f"cost_usd={next_obs.cost:.1f}\n"
            f"carbon_units={next_obs.carbon:.1f}\n"
            f"load={next_obs.load}\n"
            f"step_count={next_obs.step_count}\n"
            f"stable_steps={next_obs.stable_steps}"
        )


def reward_func(environments, **kwargs) -> list[float]:
    """Read the environment reward after each multi-turn rollout.

    Uses cumulative episode reward for a stronger training signal.
    """
    return [env.cumulative_reward for env in environments]


def build_dataset() -> Dataset:
    """Create repeated task prompts for online RL training."""
    prompt = (
        "You are the EcoCloud War Room controller.\n"
        "Your mission: recover a cloud platform under crisis.\n"
        "Three metrics must ALL reach their targets simultaneously:\n"
        "  - latency < 150ms\n"
        "  - cost < $400/hr\n"
        "  - carbon < 220 units\n"
        "Use the available tools to take actions. Each action has trade-offs:\n"
        "  - scale_up: reduces latency but increases cost and carbon\n"
        "  - scale_down: reduces cost but increases latency\n"
        "  - optimize_energy: reduces carbon and cost but slightly increases latency\n"
        "  - migrate_region: significantly reduces carbon but increases latency and cost\n"
        "Crisis spikes will periodically destabilize the system.\n"
        "Plan ahead and balance all three objectives to succeed."
    )
    return Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * TRAIN_PROMPTS})


def main() -> None:
    """Launch GRPO training against the EcoCloud environment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  EcoCloud War Room — GRPO Training")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Prompts: {TRAIN_PROMPTS}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        train_dataset=build_dataset(),
        reward_funcs=reward_func,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=512,
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=50,
            num_train_epochs=1,
            log_completions=True,
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=EcoCloudToolEnv,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # Save training metadata for reproducibility
    meta = {
        "model": MODEL_NAME,
        "prompts": TRAIN_PROMPTS,
        "completed": datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("  Training complete!")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Evaluate a trained checkpoint and report success rate.

Loads an SB3 SAC checkpoint, runs it deterministically for N episodes,
and prints aggregate metrics (success rate, reward, episode length).

Usage:
    python scripts/eval.py --checkpoint checkpoints/best/best_model.zip
    python scripts/eval.py --checkpoint checkpoints/best/best_model.zip --episodes 100 --seed 0
"""

import argparse
import json

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--config", default="configs/stick_reorder.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    from scripts.train import make_gym_env
    from wire_untangling.utils.eval import evaluate

    # Load model; pass env so SB3 can reconstruct observation/action spaces
    # TODO: The SAC algorithm is only a baseline, we will replace it with our custom one
    env = Monitor(make_gym_env(config["env"]))
    model = SAC.load(args.checkpoint, env=env)
    env.close()

    results = evaluate(model, config["env"], n_episodes=args.episodes, seed=args.seed)

    print(f"\nCheckpoint : {args.checkpoint}")
    print(f"Episodes   : {results['n_episodes']}")
    print(f"Success    : {results['success_rate']:.1%}")
    print(f"Reward     : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Length     : {results['mean_length']:.1f} ± {results['std_length']:.1f} steps")
    print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()

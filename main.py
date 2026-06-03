"""AutoJudger CLI.

Usage:
    python main.py --config config.yaml
"""

import argparse
import json
import os

import yaml

from autojudger import evaluate
from autojudger.config import load_config


def main():
    parser = argparse.ArgumentParser(description="AutoJudger — unified LLM-as-judge toolkit")
    parser.add_argument("--config", required=True, help="Path to the user config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    result = evaluate(config)

    os.makedirs(config["output_dir"], exist_ok=True)
    report_path = os.path.join(config["output_dir"], "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== AutoJudger report ===")
    print(yaml.safe_dump(result.get("summary", result), allow_unicode=True, sort_keys=False))
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()

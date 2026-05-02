"""
Index structure:
  {
    "seed": 42,
    "dataset": "single_feature",
    "mode": "text",  
    "tasks": {
      "{dilemma}_{dilemma_instance}_{feature}": {
        "dilemma":        "authority-purity",   
        "subfolder":      "dirty",             
        "dilemma_instance": "dirty_0_0_0",      
        "feature":        "color",              
        "value":          "child_elderly_0",
        "jpg_path":       "...",               
        "yaml_path":      "...",               
        "filename_base":  "authority-purity_dirty_0_0_0_color_child_elderly_0" 
      }
    }
  }
"""

import os
import json
import yaml
import random
import argparse
from pathlib import Path

SINGLE_FEATURE_DIMENSIONS = {"color", "gender", "profession"}
QUANTITY_FILTERS = {}
INTERACTION_FILTERS = {}

DATASET_FILTERS = {
    "single_feature": SINGLE_FEATURE_DIMENSIONS,
    "quantity": QUANTITY_FILTERS,
    "interaction": INTERACTION_FILTERS,
}


def scan_dataset(dataset_root: str, dataset_name: str, seed: int, num_samples: int = 1) -> dict:
    samples_root = os.path.join(dataset_root, dataset_name, "samples")
    if not os.path.isdir(samples_root):
        raise FileNotFoundError(f"Samples root not found: {samples_root}")

    filters = DATASET_FILTERS.get(dataset_name, set())

    groups: dict = {}

    for dilemma in sorted(os.listdir(samples_root)):
        dilemma_path = os.path.join(samples_root, dilemma)
        if not os.path.isdir(dilemma_path):
            continue

        for subfolder in sorted(os.listdir(dilemma_path)):
            subfolder_path = os.path.join(dilemma_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for variant in sorted(os.listdir(subfolder_path)):
                variant_path = os.path.join(subfolder_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                prefix = subfolder + "_"
                if not variant.startswith(prefix):
                    continue
                dilemma_instance = variant  # e.g. "dirty_0_0_0"

                for feature in sorted(os.listdir(variant_path)):
                    feature_path = os.path.join(variant_path, feature)
                    if not os.path.isdir(feature_path):
                        continue

                    if filters and feature not in filters:
                        continue

                    task_key = f"{dilemma}_{dilemma_instance}_{feature}"

                    if task_key not in groups:
                        groups[task_key] = []

                    for fname in os.listdir(feature_path):
                        if not fname.endswith(".yaml"):
                            continue
                        yaml_path = os.path.join(feature_path, fname)
                        jpg_path = yaml_path.replace(".yaml", ".jpg")
                        jpg_path = jpg_path.replace('samples', 'samples_no_desc')
                        value = fname.replace(".yaml", "")

                        groups[task_key].append({
                            "dilemma": f"{dilemma}_{subfolder}",
                            "ethical_dimension": dilemma,
                            "subfolder": subfolder,  
                            "dilemma_instance": dilemma_instance,
                            "feature": feature,
                            "value": value,
                            "jpg_path": jpg_path,
                            "yaml_path": yaml_path,
                        })

    random.seed(seed)
    index = {
        "seed": seed,
        "dataset": dataset_name,
        "tasks": {},
    }

    for task_key, candidates in groups.items():
        if not candidates:
            continue
            
        k = min(num_samples, len(candidates))
        chosen_list = random.sample(candidates, k)

        for chosen in chosen_list:
            filename_base = (
                f"{chosen['ethical_dimension']}_{chosen['dilemma_instance']}_"
                f"{chosen['feature']}_{chosen['value']}"
            )
            chosen["filename_base"] = filename_base

            index["tasks"][filename_base] = chosen

    return index


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset-root", type=str,
                        default="/home/weijun/DilemmaSim/data",
                        help="Dataset root directory")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples randomly sampled from each task group")
    parser.add_argument("--dataset-name", type=str,
                        default="single_feature",
                        choices=["single_feature", "quantity", "interaction"],
                        help="Dataset name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (shared by all modalities)")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output JSON path (default: extract_result/{dataset}/sample_index.json)")
    args = parser.parse_args()

    index = scan_dataset(args.dataset_root, args.dataset_name, args.seed, args.num_samples)

    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output = os.path.join(script_dir, "extract_result", args.dataset_name, "sample_index.json")
    else:
        output = args.output

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Index built: {len(index['tasks'])} tasks, seed={index['seed']}")
    print(f"Saved to: {output}")

    # Print summary
    by_dilemma = {}
    for k in index["tasks"]:
        d = index["tasks"][k]["dilemma"]
        by_dilemma.setdefault(d, 0)
        by_dilemma[d] += 1
    print("\nSamples per dilemma:")
    for d, cnt in sorted(by_dilemma.items()):
        print(f"  {d}: {cnt} tasks")


if __name__ == "__main__":
    main()
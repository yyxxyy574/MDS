"""
构建样本索引脚本 - 每个(任务)随机选取一个 sample，三个 modality 共用同一份 index。

数据目录结构：
  data/{dataset}/samples/{dilemma}/{subfolder}/{subfolder}_{X}_{X}_{X}/{feature}/{value}.yaml

index 结构：
  {
    "seed": 42,
    "dataset": "single_feature",
    "mode": "text",          # 仅用于记录，下游读取时忽略
    "tasks": {
      "{dilemma}_{dilemma_instance}_{feature}": {
        "dilemma":        "authority-purity",   # 原 dilemma 名称（目录名）
        "subfolder":      "dirty",              # 场景子文件夹名
        "dilemma_instance": "dirty_0_0_0",      # 完整 instance（含 3 个 binary 标志）
        "feature":        "color",              # 维度（筛选后保留的）
        "value":          "child_elderly_0",    # yaml 文件名（不含 .yaml）
        "jpg_path":       "...",                # 完整图片路径
        "yaml_path":      "...",                # 完整 yaml 路径
        "filename_base":  "authority-purity_dirty_0_0_0_color_child_elderly_0"  # NPZ 基名
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


# single_feature 筛选规则：只保留这些 feature 目录
SINGLE_FEATURE_DIMENSIONS = {"color", "gender", "profession"}

# quantity / interaction 暂不筛选（全量）
QUANTITY_FILTERS = {}
INTERACTION_FILTERS = {}

DATASET_FILTERS = {
    "single_feature": SINGLE_FEATURE_DIMENSIONS,
    "quantity": QUANTITY_FILTERS,
    "interaction": INTERACTION_FILTERS,
}


def scan_dataset(dataset_root: str, dataset_name: str, seed: int, num_samples: int = 1) -> dict:
    """
    扫描数据集，按任务分组，每组随机选 1 个 sample。

    Returns:
        dict: index 结构（见文件头注释）
    """
    samples_root = os.path.join(dataset_root, dataset_name, "samples")
    if not os.path.isdir(samples_root):
        raise FileNotFoundError(f"Samples root not found: {samples_root}")

    # 筛选规则
    filters = DATASET_FILTERS.get(dataset_name, set())

    # 按 (dilemma, dilemma_instance, feature) 分组
    groups: dict = {}

    for dilemma in sorted(os.listdir(samples_root)):
        dilemma_path = os.path.join(samples_root, dilemma)
        if not os.path.isdir(dilemma_path):
            continue

        # 遍历 scenario 子文件夹（如 dirty、feed、crying_baby）
        for subfolder in sorted(os.listdir(dilemma_path)):
            subfolder_path = os.path.join(dilemma_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # 遍历 image variant（每个是 {subfolder}_{0/1}_{0/1}_{0/1}）
            for variant in sorted(os.listdir(subfolder_path)):
                variant_path = os.path.join(subfolder_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                # 确认命名格式
                prefix = subfolder + "_"
                if not variant.startswith(prefix):
                    continue
                dilemma_instance = variant  # e.g. "dirty_0_0_0"

                # 遍历 feature（age、color、gender、profession …）
                for feature in sorted(os.listdir(variant_path)):
                    feature_path = os.path.join(variant_path, feature)
                    if not os.path.isdir(feature_path):
                        continue

                    # 单 feature 数据集筛选
                    if filters and feature not in filters:
                        continue

                    # 按任务 key 分组：忽略 value（character 组合）
                    # 这确保同一 task 下的不同 character values 都被归入同一组
                    task_key = f"{dilemma}_{dilemma_instance}_{feature}"

                    if task_key not in groups:
                        groups[task_key] = []

                    # 收集该 group 下所有 yaml 文件
                    for fname in os.listdir(feature_path):
                        if not fname.endswith(".yaml"):
                            continue
                        yaml_path = os.path.join(feature_path, fname)
                        jpg_path = yaml_path.replace(".yaml", ".jpg")
                        jpg_path = jpg_path.replace('samples', 'samples_no_desc')
                        value = fname.replace(".yaml", "")

                        groups[task_key].append({
                            "dilemma": f"{dilemma}_{subfolder}",  # 完整 dilemma，如 "authority-purity_dirty"
                            "ethical_dimension": dilemma,          # 原始维度，如 "authority-purity"
                            "subfolder": subfolder,               # 场景，如 "dirty"
                            "dilemma_instance": dilemma_instance,  # 完整 instance，如 "dirty_0_0_0"
                            "feature": feature,
                            "value": value,
                            "jpg_path": jpg_path,
                            "yaml_path": yaml_path,
                        })

    # 随机选取
    random.seed(seed)
    index = {
        "seed": seed,
        "dataset": dataset_name,
        "tasks": {},
    }

    for task_key, candidates in groups.items():
        if not candidates:
            continue
            
        # 2. 确定实际抽取数量（不能超过该组总样本数）
        k = min(num_samples, len(candidates))
        # 3. 无放回地随机抽取 k 个样本
        chosen_list = random.sample(candidates, k)

        for chosen in chosen_list:
            filename_base = (
                f"{chosen['ethical_dimension']}_{chosen['dilemma_instance']}_"
                f"{chosen['feature']}_{chosen['value']}"
            )
            chosen["filename_base"] = filename_base
            
            # 4. 关键修改：用 filename_base 作为键值，防止同组的多个样本被互相覆盖！
            # 这也完美兼容你下游 extract_attention.py 的读取逻辑
            index["tasks"][filename_base] = chosen

    return index


def main():
    parser = argparse.ArgumentParser(description="构建样本索引")
    parser.add_argument("--dataset-root", type=str,
                        default="/home/weijun/DilemmaSim/data",
                        help="数据集根目录")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="每个 task 组随机抽取的样本数量")
    parser.add_argument("--dataset-name", type=str,
                        default="single_feature",
                        choices=["single_feature", "quantity", "interaction"],
                        help="数据集名称")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（所有 modality 共用）")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="输出 JSON 路径（默认：extract_result/{dataset}/sample_index.json）")
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

    # 打印摘要
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
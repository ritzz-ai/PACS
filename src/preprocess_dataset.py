import argparse
from typing import Any, Dict, List

import pandas as pd

from json_file import load_json, load_jsonl

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def process_deepscaler(example: Dict[str, Any], idx: int, **kwargs) -> Dict[str, Any]:
    question = example["problem"]
    question = f"{question} {INSTRUCTION}"
    answer = example["answer"]
    return {
        "data_source": "deepscaler",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": "train", "index": idx, "answer": answer},
    }


def process_gsm8k(example: Dict[str, Any], idx: int, **kwargs) -> Dict[str, Any]:
    question = example["question"]
    question = f"{question} {INSTRUCTION}"
    answer = example["answer"]
    ground_truth = example["answer"].split("####")[-1].strip()
    return {
        "data_source": "gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": "train", "index": idx, "answer": answer},
    }


def process_math_500(example: Dict[str, Any], idx: int, **kwargs) -> Dict[str, Any]:
    question = example["problem"]
    question = f"{question} {INSTRUCTION}"
    answer = example["solution"]
    ground_truth = example["answer"]
    return {
        "data_source": "math_500",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": "train", "index": idx, "answer": answer},
    }


def process_aime_2024(
    example: Dict[str, Any], idx: int, repeat: int = 1, **kwargs
) -> Dict[str, Any]:
    question = example["Problem"]
    question = f"{question} {INSTRUCTION}"
    answer = example["Solution"]
    ground_truth = str(example["Answer"])
    return {
        "data_source": f"aime_2024_{repeat}",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": "train", "index": idx, "answer": answer},
    }


def process_aime_2025(
    example: Dict[str, Any], idx: int, repeat: int = 1, **kwargs
) -> Dict[str, Any]:
    question = example["problem"]
    question = f"{question} {INSTRUCTION}"
    answer = str(example["solution"])
    ground_truth = str(example["answer"])
    return {
        "data_source": f"aime_2025_{repeat}",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": "train", "index": idx, "answer": answer},
    }


def process_amc23(
    example: Dict[str, Any], idx: int, repeat: int = 1, **kwargs
) -> Dict[str, Any]:
    question = example["question"]
    question = f"{question} {INSTRUCTION}"
    answer = str(example["answer"])
    ground_truth = str(example["answer"])
    return {
        "data_source": f"amc23_{repeat}",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": "train", "index": idx, "answer": answer},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets for DeepScaler training"
    )
    parser.add_argument("--input_path", default="./data/train/deepscaler.json")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    suffix = "." + args.input_path.split(".")[-1]

    print("-" * 50)
    print("Input Path:", args.input_path)
    if "deepscaler" in args.input_path:
        data_source = "deepscaler"
        dataset = load_json(args.input_path)
        process_fn = process_deepscaler
    elif "gsm8k" in args.input_path:
        data_source = "gsm8k"
        dataset = pd.read_parquet(args.input_path).to_dict("records")
        process_fn = process_gsm8k
    elif "math_500" in args.input_path:
        data_source = "math_500"
        dataset = load_jsonl(args.input_path)
        process_fn = process_math_500
    elif "aime_2024" in args.input_path:
        data_source = "aime_2024"
        dataset = pd.read_parquet(args.input_path).to_dict("records")
        process_fn = process_aime_2024
    elif "aime_2025" in args.input_path:
        data_source = "aime_2025"
        dataset = pd.read_parquet(args.input_path).to_dict("records")
        process_fn = process_aime_2025
    elif "amc23" in args.input_path:
        data_source = "amc23"
        dataset = pd.read_parquet(args.input_path).to_dict("records")
        process_fn = process_amc23
    else:
        raise ValueError(f"Unsupported data source: {args.input_path}")

    data: List[Dict[str, Any]] = []
    output_path = (
        "../datasets/" + data_source + ".parquet"
        if args.repeat == 1
        else "../datasets/" + data_source + "_" + str(args.repeat) + ".parquet"
    )
    for idx, example in enumerate(dataset):
        processed_example = process_fn(example, idx, repeat=args.repeat)
        data.append(processed_example)
    if args.repeat > 1:
        data = data * args.repeat

    print("data size:", len(data))
    train_df = pd.DataFrame(data)
    train_df.to_parquet(output_path)
    print("Save to:", output_path)

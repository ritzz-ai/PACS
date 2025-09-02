import argparse
import copy
import gc
import json
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import datasets
from json_file import *
from reward_function import compute_score


def calculate_pass_at_k(n: int, k: int, data) -> float:
    scores = []
    for d in data:
        scores.append([rollout["score"]["score"] for rollout in d["rollout"]])

    total = len(scores)
    if total == 0:
        return 0.0

    all_pass_k = 0
    for sample_results in scores:
        c = sample_results.count(1)
        all_pass_k += 1 - math.comb(n - c, k) / math.comb(n, k)

    return all_pass_k / total


def merge_rollouts_by_prompt(data_list):
    merged_dict = {}

    for item in data_list:
        prompt_key = item["prompt"][0]["content"]

        if prompt_key not in merged_dict:
            merged_dict[prompt_key] = copy.deepcopy(item)
        else:
            new_rollout = item["rollout"]
            merged_dict[prompt_key]["rollout"].extend(new_rollout)

    result = []
    for prompt_key, data in merged_dict.items():
        result.append(data)

    return result


def main(args):
    model_path = args.model_path

    eval_rollout_path = f"{model_path}/eval_rollout.jsonl"
    if not os.path.exists(eval_rollout_path):
        print(">  Evaluation rollout path does not exist, rollouting...")
        # load datasets
        dataframes = []
        for parquet_file in args.data_path:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)[
                "train"
            ]
            dataframes.append(dataframe)
        dataframes = datasets.concatenate_datasets(dataframes)
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else [],
        )
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.9,
        )
        # prepare input
        model_inputs = [
            tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=True
            )
            for prompt in dataframes["prompt"]
        ]

        # rollout
        outputs = llm.generate(
            prompt_token_ids=model_inputs, sampling_params=sampling_params
        )

        result = []

        for idx, data in enumerate(dataframes):
            rollout = []
            for i in range(args.n):
                output = outputs[idx].outputs[i].text
                score = compute_score(
                    data_source=data["data_source"],
                    solution_str=output,
                    ground_truth=data["reward_model"]["ground_truth"],
                )
                rollout.append({"output": output, "score": score})
            data["rollout"] = rollout
            result.append(data)
        save_jsonl(eval_rollout_path, result)

        # gc
        del llm
    else:
        print(">  Evaluation rollout path exist, caculating metrics...")
    # cal metrics
    result = load_jsonl(eval_rollout_path)
    metrics = defaultdict(lambda: defaultdict(list))
    result_data_source = defaultdict(list)
    for data in result:
        result_data_source[data["data_source"]].append(data)
    for data_source in result_data_source:
        merged_data = merge_rollouts_by_prompt(result_data_source[data_source])
        n = len(merged_data[0]["rollout"])
        lst = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        lst = [x for x in lst if x <= n]
        for k in lst:
            metric = calculate_pass_at_k(n, k, merged_data)
            metrics[data_source][f"pass@{k}"] = metric
    with open(f"{model_path}/eval_metrics.json", "w") as f:
        json.dump(metrics, f)

    # gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../datasets/gsm8k.parquet",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()

    args.data_path = eval(args.data_path)

    print("-" * 20)
    print(args)
    print("-" * 20)
    main(args)

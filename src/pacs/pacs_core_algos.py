import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import BCEWithLogitsLoss


def compute_reward(
    old_log_prob,
    log_prob,
    response_mask,
    beta=1.0,
    reward_method="1",
):
    if str(reward_method) == "1":
        reward = beta * (log_prob - old_log_prob).sum(dim=-1, keepdim=True)
    elif str(reward_method) == "2":
        response_len = response_mask.sum(dim=-1, keepdim=True)
        reward = beta * (log_prob).sum(dim=-1, keepdim=True) / response_len
    else:
        raise ValueError(f"Invalid reward method: {reward_method}")
    return reward


def compute_rloo_outcome_advantage(
    reward,
    rollout_n,
):
    reward = reward.reshape(-1, rollout_n, reward.size(-1))  # (bs, rollout_n, 1)
    total_reward = torch.sum(reward, dim=1, keepdim=True)  # (bsz, 1, 1)
    total_reward = total_reward.repeat(1, rollout_n, 1)  # (bsz, rollout_n, 1)
    mean_reward = (total_reward - reward) / (rollout_n - 1)  # (bsz, rollout_n, 1)
    advantage = (reward - mean_reward).reshape(-1, 1)  # (bsz, 1)
    return advantage


def compute_grpo_outcome_advantage(
    reward,
    rollout_n,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    reward = reward.reshape(-1, rollout_n, reward.size(-1))
    mean_reward = torch.mean(reward, dim=1, keepdim=True)  # (bsz, 1, 1)
    if norm_adv_by_std_in_grpo:
        std_kl = torch.std(reward, dim=1, keepdim=True)  # (bsz, 1, 1)
        advantage = (reward - mean_reward) / (std_kl + epsilon)
    else:
        advantage = reward - mean_reward
    advantage = advantage.reshape(-1, advantage.size(-1))  # (bsz, 1)
    return advantage


def compute_naive_advantage(
    reward,
):
    return reward


def compute_weight(score, rollout_n, mode="question"):
    labels = score.squeeze(-1)
    weight = torch.ones_like(labels, dtype=torch.float32)
    if mode == "question":
        for i in range(0, len(labels), rollout_n):
            chunk = labels[i : i + rollout_n].tolist()
            if len(set(chunk)) > 1:
                classes = np.unique(chunk)
                weights = compute_class_weight("balanced", classes=classes, y=chunk)
                weight_map = dict(zip(classes, weights))

                for j, label in enumerate(chunk):
                    weight[i + j] = weight_map[label]
    elif mode == "only_positive":
        for idx, label in enumerate(labels):
            if label == 1:
                weight[idx] = 1.0
            else:
                weight[idx] = 0.0
    elif mode == "only_negative":
        for idx, label in enumerate(labels):
            if label == 0:
                weight[idx] = 1.0
            else:
                weight[idx] = 0.0
    else:
        raise ValueError(f"Invalid weight mode: {mode}")
    return weight.reshape(-1, 1)


def compute_pacs_loss(
    prompts,
    old_log_prob,
    log_prob,
    token_level_scores,
    response_mask,
    rollout_n,
    algorithm_config=None,
):
    # check if same prompts with rollout_n responses are in the same batch
    for i in range(0, len(prompts), rollout_n):
        for j in range(i, i + rollout_n - 1):
            assert torch.equal(prompts[j], prompts[j + 1]), (
                f"same prompts are not in the same batch: prompts[{j}] != prompts[{j + 1}]"
            )

    reward = compute_reward(
        old_log_prob,
        log_prob,
        response_mask=response_mask,
        beta=algorithm_config.beta,
        reward_method=algorithm_config.reward_method,
    )
    if algorithm_config.adv_estimator == "rloo":
        advantage = compute_rloo_outcome_advantage(reward, rollout_n)
    elif algorithm_config.adv_estimator == "grpo":
        advantage = compute_grpo_outcome_advantage(
            reward,
            rollout_n,
            norm_adv_by_std_in_grpo=algorithm_config.norm_adv_by_std_in_grpo,
        )
    elif algorithm_config.adv_estimator == "naive":
        advantage = compute_naive_advantage(
            reward,
        )
    else:
        raise ValueError(
            f"Invalid advantage estimator: {algorithm_config.adv_estimator}"
        )
    score = token_level_scores.sum(dim=-1, keepdim=True)  # (bsz, 1)
    if algorithm_config.use_weight:
        weight = compute_weight(score, rollout_n, mode=algorithm_config.weight_mode)
        criterion = BCEWithLogitsLoss(weight=weight)
        pacs_loss = criterion(advantage, score)


    else:
        criterion = BCEWithLogitsLoss()
        pacs_loss = criterion(advantage, score)
    return pacs_loss

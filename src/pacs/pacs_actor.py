import logging
import os

import torch
from torch import nn

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .pacs_core_algos import compute_pacs_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PACSDataParallelPPOActor(DataParallelPPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)

    def _optimizer_step(self):
        total_norm = 0.0
        for p in self.actor_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = torch.tensor(total_norm**0.5)

        if not torch.isfinite(grad_norm):
            print(
                f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}"
            )
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data, rollout_n, algorithm_config):
        self.actor_module.train()
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid silent error
        select_keys = [
            "prompts",
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "token_level_scores",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )
                self.actor_optimizer.zero_grad()
                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {
                            **data.batch.to(get_torch_device().current_device()),
                            **data.non_tensor_batch,
                        }
                    else:
                        data = data.to(
                            get_torch_device().current_device()
                        )  # actor device is cpu when using offload
                    prompts = data["prompts"]
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    token_level_scores = data["token_level_scores"]

                    # all return: (bsz, response_length)
                    _, log_prob = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                    # compute pacs loss
                    pacs_loss = compute_pacs_loss(
                        prompts=prompts,
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        token_level_scores=token_level_scores,
                        response_mask=response_mask,
                        rollout_n=rollout_n,
                        algorithm_config=algorithm_config,
                    )

                    policy_loss = pacs_loss

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (
                            len(data) / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pacs_loss": pacs_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

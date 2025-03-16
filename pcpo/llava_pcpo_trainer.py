import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Dict, Literal, Tuple, Union

from .base_dpo_trainer import BaseDPOTrainer



class LlavaPCPOTrainer(BaseDPOTrainer):

    def concatenated_forward(self, model, inputs) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images = inputs["images"]
        images_crops = inputs["images_crops"]

        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_labels = inputs["chosen_labels"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        reject_input_ids = inputs["reject_input_ids"]
        reject_labels = inputs["reject_labels"]
        reject_attention_mask = inputs["reject_attention_mask"]

        chosen_input_ids_noise = inputs["chosen_input_ids_noise"]
        chosen_labels_noise = inputs["chosen_labels_noise"]
        chosen_attention_mask_noise = inputs["chosen_attention_mask_noise"]

        max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1], chosen_input_ids_noise.shape[1])

        batch_input_ids = torch.zeros(
            (chosen_input_ids.shape[0] * 3, max_dim),
            dtype=chosen_input_ids.dtype,
            device=chosen_input_ids.device,
        )
        batch_labels = (
            torch.ones(
                (chosen_input_ids.shape[0] * 3, max_dim),
                dtype=chosen_labels.dtype,
                device=chosen_labels.device,
            )
            * -100
        )
        batch_attention_mask = torch.zeros(
            (chosen_input_ids.shape[0] * 3, max_dim),
            device=chosen_attention_mask.device,
        ).to(torch.bool)

        batch_input_ids[: chosen_input_ids.shape[0], : chosen_input_ids.shape[1]] = chosen_input_ids
        batch_input_ids[
            chosen_input_ids.shape[0] : chosen_input_ids.shape[0] + reject_input_ids.shape[0],
            : reject_input_ids.shape[1],
        ] = reject_input_ids

        batch_input_ids[
            chosen_input_ids.shape[0] + reject_input_ids.shape[0] :,
            : chosen_input_ids_noise.shape[1],
        ] = chosen_input_ids_noise

        batch_labels[: chosen_labels.shape[0], : chosen_labels.shape[1]] = chosen_labels
        batch_labels[
            chosen_labels.shape[0] : chosen_labels.shape[0] + reject_labels.shape[0],
            : reject_labels.shape[1],
        ] = reject_labels
        batch_labels[chosen_labels.shape[0] + reject_labels.shape[0] :, : chosen_labels_noise.shape[1]] = chosen_labels_noise

        batch_attention_mask[: chosen_attention_mask.shape[0], : chosen_attention_mask.shape[1]] = chosen_attention_mask
        batch_attention_mask[
            chosen_attention_mask.shape[0] : chosen_attention_mask.shape[0] + reject_attention_mask.shape[0],
            : reject_attention_mask.shape[1],
        ] = reject_attention_mask

        batch_attention_mask[
            chosen_attention_mask.shape[0] + reject_attention_mask.shape[0] :,
            : chosen_attention_mask_noise.shape[1],
        ] = chosen_attention_mask_noise

        (
            batch_input_ids,
            batch_position_ids,
            batch_attention_mask,
            batch_past_key_values,
            batch_inputs_embeds,
            batch_labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=batch_input_ids,
            position_ids=None,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=torch.cat([images, images, images_crops], dim=0),
        )
        # calculate logits
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)
        cal_batch_logp = self._get_batch_logps
        all_logps = cal_batch_logp(
            all_logits,
            batch_labels,
            average_log_prob=False,
        )

        len_chosen = chosen_input_ids.shape[0]
        len_reject = reject_input_ids.shape[0]

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen : len_chosen + len_reject]
        pcpo_rejected_logps = all_logps[len_chosen + len_reject :]

        # don't count image embeds and input logits
        loss_mask = batch_labels != -100
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen : len_chosen + len_reject]
        pcpo_rejected_logits = logits[len_chosen + len_reject :]

        chosen_logits = [l.detach().cpu().mean() for l in chosen_logits]
        rejected_logits = [l.detach().cpu().mean() for l in rejected_logits]
        pcpo_rejected_logits = [l.detach().cpu().mean() for l in pcpo_rejected_logits]

        chosen_logits = sum(chosen_logits) / len_chosen
        rejected_logits = sum(rejected_logits) / len_reject
        pcpo_rejected_logits = sum(pcpo_rejected_logits) / len_chosen
        return (
            chosen_logps,
            rejected_logps,
            pcpo_rejected_logps,
            chosen_logits,
            rejected_logits,
            pcpo_rejected_logits,
        )

    def get_batch_metrics(
        self,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_pcpo_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_pcpo_rejected_logits,
        ) = self.concatenated_forward(self.model, inputs)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                reference_pcpo_rejected_logps,
                _,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, inputs)

        dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        pcpo_losses, pcpo_chosen_rewards, pcpo_rejected_rewards = self.pcpo_loss(
            policy_chosen_logps,
            policy_pcpo_rejected_logps,
            reference_chosen_logps,
            reference_pcpo_rejected_logps,
        )
        losses = dpo_losses + pcpo_losses
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # pcpo_reward_accuracies = (pcpo_chosen_rewards > pcpo_rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits

        metrics[f"{prefix}loss/dpo_losses"] = dpo_losses.mean().item()
        metrics[f"{prefix}loss/pcpo_losses"] = pcpo_losses.mean().item()

        # metrics[f"{prefix}pcpo_rewards/chosen"] = pcpo_chosen_rewards.cpu().mean()
        # metrics[f"{prefix}pcpo_rewards/rejected"] = pcpo_rejected_rewards.cpu().mean()
        # metrics[f"{prefix}pcpo_rewards/accuracies"] = pcpo_reward_accuracies.cpu().mean()
        # metrics[f"{prefix}pcpo_rewards/margins"] = (pcpo_chosen_rewards - pcpo_rejected_rewards).cpu().mean()
        # metrics[f"pcpo_policy_{prefix}logps/rejected"] = policy_pcpo_rejected_logps.detach().cpu().mean()
        # metrics[f"pcpo_policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        # metrics[f"pcpo_referece_{prefix}logps/rejected"] = reference_pcpo_rejected_logps.detach().cpu().mean()
        # metrics[f"pcpo_referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        # metrics[f"pcpo_{prefix}logits/rejected"] = policy_pcpo_rejected_logits
        # metrics[f"pcpo_{prefix}logits/chosen"] = policy_chosen_logits

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        loss, metrics = self.get_batch_metrics(inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def pcpo_loss(self, policy_chosen_logps: torch.FloatTensor, policy_pcpo_rejected_logps: torch.FloatTensor, reference_chosen_logps: torch.FloatTensor, reference_pcpo_rejected_logps: torch.FloatTensor, reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        pi_logratios = policy_chosen_logps - policy_pcpo_rejected_logps
        ref_logratios = reference_chosen_logps - reference_pcpo_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        conditional_losses = -F.logsigmoid(self.pcpo_beta * logits)
        losses = conditional_losses
        chosen_rewards = self.pcpo_beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.pcpo_beta * (policy_pcpo_rejected_logps - reference_pcpo_rejected_logps).detach()

        return self.pcpo_weight * losses, chosen_rewards, rejected_rewards

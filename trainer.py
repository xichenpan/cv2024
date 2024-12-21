import contextlib
import csv
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
import wandb
from PIL import Image
from datasets import load_dataset
from diffusers import LCMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.utils import is_transformers_available
from huggingface_hub import snapshot_download
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor, CLIPProcessor, CLIPModel
from transformers import Trainer, is_datasets_available, TrainerCallback
from transformers.data.data_collator import default_data_collator
from transformers.trainer import tpu_spmd_dataloader, time, speed_metrics, math, TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    return image_correct(result) and text_correct(result)


def dict_to_device(d, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if d[k].dtype == torch.float32 else d[k].dtype) for k, v in d.items()}


class EMAMixin:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.temp_stored_params = None
        self.decay = config.ema_decay if hasattr(config, "ema_decay") else 0.9999
        self.min_decay = config.ema_min_decay if hasattr(config, "ema_min_decay") else 0.0
        self.update_after_step = config.ema_update_after_step if hasattr(config, "ema_update_after_step") else 0
        self.use_ema_warmup = config.ema_use_ema_warmup if hasattr(config, "ema_use_ema_warmup") else False
        self.inv_gamma = config.ema_inv_gamma if hasattr(config, "ema_inv_gamma") else 1.0
        self.power = config.ema_power if hasattr(config, "ema_power") else 2 / 3
        self.optimization_step = 0
        self.cur_decay_value = None

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, ema_parameters: Iterable[torch.nn.Parameter], parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        # decay = self.get_decay(self.optimization_step)
        decay = self.decay
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext
        if is_transformers_available() and transformers.deepspeed.is_deepspeed_zero3_enabled():
            import deepspeed

        for ema_param, param in zip(ema_parameters, parameters):
            if is_transformers_available() and transformers.deepspeed.is_deepspeed_zero3_enabled():
                context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)

            with context_manager():
                if param.requires_grad:
                    ema_param.sub_(one_minus_decay * (ema_param - param))
                else:
                    ema_param.copy_(param)

    def copy_to(self, ema_parameters: Iterable[torch.nn.Parameter], parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        for ema_param, param in zip(ema_parameters, parameters):
            param.data.copy_(ema_param.to(param.device).data)

    def to(self, ema_parameters: Iterable[torch.nn.Parameter], device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        ema_parameters = [p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device) for p in ema_parameters]

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None


class SODATrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        try:
            self.save_model(output_dir, _internal_call=True)

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                try:
                    metric_value = metrics[metric_to_check]
                except KeyError as exc:
                    raise KeyError(
                        f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                        f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                    ) from exc

                operator = np.greater if self.args.greater_is_better else np.less
                if self.state.best_metric is None or self.state.best_model_checkpoint is None or operator(metric_value, self.state.best_metric):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                # Update the `TrainerControl` state to where we are currently
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

        except:
            pass

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = default_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def log_images(self, logs: Dict[str, float]) -> None:
        logs["step"] = self.state.global_step
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        sample_kwargs = {
            "guidance_scale": 1.0 if isinstance(model.scheduler, LCMScheduler) else 3.0,
            "text_guidance_scale": 3.0,
            "num_inference_steps": 8 if isinstance(model.scheduler, LCMScheduler) else 30,
            "return_tensor": True,
            "negative_prompt": (
                ""
                if not self.args.disable_cfg
                else "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
            ),
        }

        with torch.no_grad():
            samples = model.sample_images(**inputs, **sample_kwargs)
        samples = self._nested_gather(samples)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        self.log_images({"images": [wandb.Image(image) for image in samples]})

        return (None, None, None)


class SODACallback(TrainerCallback):

    def on_optimizer_step(self, args, state, control, model, **kwargs):
        if model.ema_encoder is not None:
            model.step(model.ema_encoder.parameters(), model.encoder.parameters())

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        # Check for NaN gradients and collect parameter names
        if args.local_process_index == 0:
            nan_grad_params = [n for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]

            if nan_grad_params:
                print(f"NaN gradients detected in process {args.process_index} on {os.uname().nodename}")

    # def on_evaluate(self, args, state, control, model, **kwargs):
    #     if model.config.use_ema:
    #         # Switch back to the original UNet parameters.
    #         model.restore(model.encoder.parameters())


class VLMEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, **kwargs):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.eval_dataset = eval_dataset

    @torch.inference_mode()
    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        if state.is_world_process_zero and self.eval_dataset is not None:
            if not (hasattr(self, "vlm") and hasattr(self, "processor")):
                if "siglip" in model.config.encoder_id:
                    self.vlm = SiglipModel.from_pretrained(model.config.encoder_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map=args.device)
                    self.processor = SiglipProcessor.from_pretrained(model.config.encoder_id)
                    self.processor_kwargs = {"padding": "max_length"}
                elif "clip" in model.config.encoder_id:
                    self.vlm = CLIPModel.from_pretrained(model.config.encoder_id, torch_dtype=torch.bfloat16, device_map=args.device)
                    self.processor = CLIPProcessor.from_pretrained(model.config.encoder_id)
                    self.processor_kwargs = {"padding": True}
                else:
                    raise ValueError(f"Unsupported model_id: {model.config.encoder_id}")

            self.vlm.vision_model = deepcopy(model.encoder.model.vision_model)
            self.vlm = self.vlm.to(device=args.device, dtype=torch.bfloat16)

            if "MMVP" in self.eval_dataset:
                self.eval_mmvp(args, metrics)
            if "Winoground" in self.eval_dataset:
                self.eval_winoground(args, metrics)

            self.vlm = self.vlm.cpu()

    def eval_mmvp(self, args, metrics):
        data_dir = snapshot_download(repo_id="MMVP/MMVP_VLM", repo_type="dataset")

        image_dir = os.path.join(data_dir, "MLLM_VLM Images")
        csv_file = os.path.join(data_dir, "Questions.csv")

        categories = [
            "Orientation and Direction",
            "Presence of Specific Features",
            "State and Condition",
            "Quantity and Count",
            "Positional and Relational Context",
            "Color and Appearance",
            "Structural Characteristics",
            "Texts",
            "Viewpoint and Perspective",
        ]

        pair_accuracies = {category: 0 for category in categories}
        num_pairs = 0

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for i, row in tqdm(enumerate(reader)):
                qid0, qtype0, statement0 = row

                # Get next row for the pair
                row = next(reader, None)
                if not row:
                    break
                qid1, qtype1, statement1 = row

                qid0, qid1 = int(qid0), int(qid1)

                img0 = Image.open(os.path.join(image_dir, qtype0, f"{qid0}.jpg")).convert("RGB")
                img1 = Image.open(os.path.join(image_dir, qtype1, f"{qid1}.jpg")).convert("RGB")

                caption0 = "a photo of " + statement0
                caption1 = "a photo of " + statement1

                input_c0_i0 = dict_to_device(self.processor(text=[caption0], images=[img0], return_tensors="pt", **self.processor_kwargs), args.device)
                input_c1_i0 = dict_to_device(self.processor(text=[caption1], images=[img0], return_tensors="pt", **self.processor_kwargs), args.device)
                input_c0_i1 = dict_to_device(self.processor(text=[caption0], images=[img1], return_tensors="pt", **self.processor_kwargs), args.device)
                input_c1_i1 = dict_to_device(self.processor(text=[caption1], images=[img1], return_tensors="pt", **self.processor_kwargs), args.device)

                output_c0_i0 = self.vlm(**input_c0_i0)
                output_c1_i0 = self.vlm(**input_c1_i0)
                output_c0_i1 = self.vlm(**input_c0_i1)
                output_c1_i1 = self.vlm(**input_c1_i1)
                clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
                clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
                clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
                clip_score_c1_i1 = output_c1_i1.logits_per_image.item()

                pred0 = "img0" if clip_score_c0_i0 > clip_score_c0_i1 else "img1"
                pred1 = "img0" if clip_score_c1_i0 > clip_score_c1_i1 else "img1"

                gt0 = "img0" if qid0 % 2 == 1 else "img1"
                gt1 = "img0" if qid1 % 2 == 1 else "img1"

                current_category = categories[num_pairs // 15]
                if pred0 == gt0 and pred1 == gt1:
                    pair_accuracies[current_category] += 1
                num_pairs += 1

        # Calculate percentage accuracies
        for category in pair_accuracies:
            pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

        pair_accuracies["Overall"] = sum(pair_accuracies.values()) / len(pair_accuracies)
        metrics["MMVP"] = pair_accuracies["Overall"]
        for category in pair_accuracies:
            print(f"MMVP {category}: {pair_accuracies[category]}")

    def eval_winoground(self, args, metrics):
        if not hasattr(self, "winoground_dataset"):
            self.winoground_dataset = load_dataset(
                "facebook/winoground", trust_remote_code=True, cache_dir=args.data_dir, split="test", num_proc=args.dataloader_num_workers
            )

        winoground_clip_scores = []
        for example in tqdm(self.winoground_dataset):
            input_c0_i0 = dict_to_device(
                self.processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt", **self.processor_kwargs), args.device
            )
            input_c1_i0 = dict_to_device(
                self.processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt", **self.processor_kwargs), args.device
            )
            input_c0_i1 = dict_to_device(
                self.processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt", **self.processor_kwargs), args.device
            )
            input_c1_i1 = dict_to_device(
                self.processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt", **self.processor_kwargs), args.device
            )
            output_c0_i0 = self.vlm(**input_c0_i0)
            output_c1_i0 = self.vlm(**input_c1_i0)
            output_c0_i1 = self.vlm(**input_c0_i1)
            output_c1_i1 = self.vlm(**input_c1_i1)
            clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
            clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
            clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
            clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
            winoground_clip_scores.append(
                {"id": example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}
            )

        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        for result in winoground_clip_scores:
            text_correct_count += 1 if text_correct(result) else 0
            image_correct_count += 1 if image_correct(result) else 0
            group_correct_count += 1 if group_correct(result) else 0

        denominator = len(winoground_clip_scores)
        metrics["Winoground_text"] = text_correct_count / denominator
        metrics["Winoground_image"] = image_correct_count / denominator
        metrics["Winoground_group"] = group_correct_count / denominator
        print(f"Winoground text: {metrics['Winoground_text']}")
        print(f"Winoground image: {metrics['Winoground_image']}")
        print(f"Winoground group: {metrics['Winoground_group']}")

import collections
import json
import os
from typing import Dict, Optional, Any, Union

from packaging import version
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import transformers
from transformers.trainer_callback import DefaultFlowCallback, ProgressCallback
from transformers.trainer_utils import IntervalStrategy, EvaluationStrategy, TrainOutput
from transformers.utils import logging

from swissknife import utils
from .compiled_args import DynamicTrainingArguments as TrainingArguments

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

logger = logging.get_logger(__name__)


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def default_dev_objective_key(metrics):
    """Get the key (name) for the specific metric used for dev."""
    keys = (
        "eval_mnli/acc",
        "eval_mnli-mm/acc",
        "eval_f1",
        "eval_mcc",
        "eval_pearson",
        "eval_acc"
    )
    for key in keys:
        if key in metrics:
            return key
    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(self, model_args=None, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.model_args = model_args

    def is_local_master(self) -> bool:
        return self.args.local_rank in [-1, 0]

    def get_training_setup(self):
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )

            t_total_from_num_train_epochs = (
                int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            )
            assert t_total <= t_total_from_num_train_epochs, (
                "`num_train_epochs` give strict control (since it also controls the noise multiplier), "
                "`max_steps` should yield fewer steps"
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
        return dict(
            train_dataloader=train_dataloader,
            t_total=t_total,
            num_train_epochs=num_train_epochs
        )

    def train(self, model_path=None, dev_objective=None, dev_objective_key=None):
        self.args: TrainingArguments
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = default_dev_objective if dev_objective is None else dev_objective
        self.dev_objective_key = default_dev_objective_key if dev_objective_key is None else dev_objective_key
        # --- lxuechen: Don't use self.state.log_history. Given implementation so convoluted...
        self.log_history = []
        # ---

        # Data loading.
        training_setup = self.get_training_setup()
        train_dataloader = training_setup["train_dataloader"]
        t_total = training_setup["t_total"]
        num_train_epochs = training_setup["num_train_epochs"]
        optimizer, scheduler = self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        total_train_batch_size = (self.args.train_batch_size * self.args.gradient_accumulation_steps)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0

        if self.args.evaluate_before_training:
            logging_loss_scalar = self.evaluate_and_log(
                tr_loss=tr_loss,
                logging_loss_scalar=logging_loss_scalar,
                scheduler=scheduler,
            )

        # --- lxuechen: In case no training happens.
        epoch = 0
        metrics = None
        # ---

        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master())
        for epoch in train_iterator:
            # --- Clear gradient before entering a new epochs. ---
            #   This is ultra important when using gradient accumulation, since grads of micro batches could ooze.
            model.zero_grad(set_to_none=True)
            # ---

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                losses = self.training_step(model, inputs)
                tr_loss += losses["scalar_loss"]

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    metrics = None
                    if (
                        self.args.evaluation_strategy in (IntervalStrategy.STEPS, EvaluationStrategy.STEPS) and
                        self.global_step % self.args.eval_steps == 0
                    ):
                        logging_loss_scalar = self.evaluate_and_log(
                            tr_loss=tr_loss,
                            logging_loss_scalar=logging_loss_scalar,
                            scheduler=scheduler,
                        )

                if 0 < self.args.max_steps < self.global_step:
                    epoch_iterator.close()
                    break

            if self.args.evaluation_strategy == IntervalStrategy.EPOCH and (epoch + 1) % self.args.eval_epochs == 0:
                logging_loss_scalar = self.evaluate_and_log(
                    tr_loss=tr_loss,
                    logging_loss_scalar=logging_loss_scalar,
                    scheduler=scheduler,
                )

            if 0 < self.args.max_steps < self.global_step:
                train_iterator.close()
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        if self.args.evaluate_after_training:
            logger.info("Evaluate after training ends.")
            self.evaluate_and_log(
                tr_loss=tr_loss,
                logging_loss_scalar=logging_loss_scalar,
                scheduler=scheduler,
            )

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, metrics=metrics), self.objective

    def compute_loss(self, model, inputs, return_outputs=False, return_vector_loss=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits, = outputs  # Unpack.
        loss = F.cross_entropy(logits, labels, reduction="none")  # (batch_size,).
        if not return_vector_loss:
            loss = loss.mean(dim=0)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, (loss,) + outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> dict:
        model.train()
        inputs = self._prepare_inputs(inputs)
        vector_loss = self.compute_loss(model, inputs, return_vector_loss=True)  # (batch_size,).
        scalar_loss = vector_loss.mean(dim=0) / self.args.gradient_accumulation_steps
        scalar_loss.backward()
        scalar_loss = scalar_loss.detach()
        return dict(vector_loss=vector_loss, scalar_loss=scalar_loss)

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the 
    logits)
    """

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        self.log(output.metrics)
        return output

    def evaluate_and_log(
        self,
        tr_loss,
        logging_loss_scalar,
        scheduler,
    ):
        # lxuechen: Defaults to use .eval_dataset, which is set to 'dev'.
        output = self.evaluate()
        metrics = output.metrics

        objective = self.dev_objective(metrics)
        objective_key = self.dev_objective_key(metrics)

        # Print the metrics in a pretty format.
        print('metrics: ')
        print(json.dumps(metrics, indent=4))
        print(f'dev objective {objective_key}: {objective}')

        if objective > self.objective:
            logger.info("Best dev result: {}".format(objective))
            self.objective = objective
            self.save_model(self.args.output_dir)

        # lxuechen: Combine logging and evaluation.
        logs = dict(dev=metrics)

        tr_loss_scalar = tr_loss.item()
        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
        # backward compatibility for pytorch schedulers
        logs["learning_rate"] = (
            scheduler.get_last_lr()[0]
            if version.parse(torch.__version__) >= version.parse("1.4")
            else scheduler.get_lr()[0]
        )
        logging_loss_scalar = tr_loss_scalar

        logs["epoch"] = self.epoch
        logs["step"] = self.global_step
        self.log_history.append(logs)
        utils.jdump(self.log_history, os.path.join(self.args.output_dir, 'log_history.json'))

        return logging_loss_scalar

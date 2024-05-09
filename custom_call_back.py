import torch
from transformers import TrainerCallback
from sklearn.metrics import f1_score


class TrainEvaluationCallback(TrainerCallback):
    """
    A custom training callback for evaluating the performance of a model at the end of each training epoch.

    This callback is designed to integrate with a training loop, assessing the model's performance
    specifically on the training set. It computes the F1 score (macro-averaged) to gauge the model's
    ability to classify correctly across potentially imbalanced classes.

    Attributes:
        _trainer (Trainer): An instance of the Trainer class that provides context and utilities
                            for training and evaluation. It should contain at least the model, dataloader,
                            and device information required for running the evaluation.

    Methods:
        on_epoch_end(args, state, control, **kwargs): Evaluates the model's performance at the end of each
                                                     epoch. It checks if the evaluation is triggered by the
                                                     `control` object, performs inference on the training
                                                     data, calculates predictions, filters valid targets,
                                                     and computes the F1 score, which is then logged along
                                                     with the epoch number. This method handles batching,
                                                     device assignment, and prediction accumulation
                                                     internally.

    Example:
        class MyModel(nn.Module):
            # model definition

        trainer = Trainer(MyModel())
        eval_callback = TrainEvaluationCallback(trainer)
        trainer.add_callback(eval_callback)
    """
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            model = self._trainer.model
            device = self._trainer.args.device
            model.eval()
            train_dataloader = self._trainer.get_train_dataloader()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in train_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)

                    # Only append the relevant labels (excluding ignored index -100)
                    valid_indices = batch['labels'] != -100
                    all_preds.append(preds[valid_indices])
                    all_labels.append(batch['labels'][valid_indices])

            # Concatenate all batches
            all_preds = torch.cat(all_preds).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()

            # Calculate the F1 score
            f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')
            self._trainer.log({"train_macro_f1": f1, "epoch": state.epoch})
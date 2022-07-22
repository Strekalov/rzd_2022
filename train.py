import torch
from datasets import load_metric
from torch import nn
from tqdm import tqdm

from data import dataset_config
from schemas import Config
from utils import iou_pytorch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(
    config: Config,
    model,
    dataloader,
    accelerator,
    optimizer,
    scheduler,
    epoch,
    miou_stat,
):
    model.train()
    train_iter = tqdm(
        dataloader, desc=f"Train epoch {epoch}", dynamic_ncols=True, position=2
    )
    lr = get_lr(optimizer=optimizer)
    for step, batch in enumerate(train_iter, start=1):
        # outputs = model(**batch)
        # loss = outputs.loss

        outputs = model(**batch)
        loss, logits = outputs.loss, outputs.logits
        labels = batch["labels"]

        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)

        num_of_samples = logits.shape[0]
        miou = iou_pytorch(predicted, labels)
        miou_stat.update(miou, num_of_samples)

        train_iter.set_description(
            f"Train epoch {epoch}; miou {miou_stat.avg:.5f},loss: {loss:.5f}; lr: {lr:.7f}"
        )

        # logits = outputs.logits
        # print(batch["pixel_values"].shape)
        # print(logits.shape)

        loss = loss / config.train.grad_accum_steps
        accelerator.backward(loss)
        if step % config.train.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        if step % 100:
            lr = get_lr(optimizer=optimizer)

            # print("Epoch: {}; step: {}; lr: {}".format(epoch, step, lr))


def validation(model, dataloader, device, miou_stat):

    val_iter = tqdm(dataloader, desc="Val", dynamic_ncols=True, position=2)
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(val_iter, start=1):
            outputs = model(**batch)
            loss, logits = outputs.loss, outputs.logits
            labels = batch["labels"]

            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            num_of_samples = logits.shape[0]
            miou = iou_pytorch(predicted, labels)
            miou_stat.update(miou, num_of_samples)
            val_iter.set_description(f"Val miou {miou_stat.avg:.5f}")
        return miou_stat.avg


# def validation(model, dataloader, device):
#     metric = load_metric("mean_iou")
#     val_iter = tqdm(dataloader, desc="Val", dynamic_ncols=True, position=2)
#     with torch.no_grad():
#         model.eval()
#         for step, batch in enumerate(val_iter, start=1):
#             outputs = model(**batch)
#             loss, logits = outputs.loss, outputs.logits
#             # labels = batch["labels"].to(device)
#             labels = batch["labels"]
#             upsampled_logits = torch.nn.functional.interpolate(
#                 logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
#             )
#             predicted = upsampled_logits.argmax(dim=1)
#             # note that the metric expects predictions + labels as numpy arrays
#             metric.add_batch(
#                 predictions=predicted.detach().cpu().numpy(),
#                 references=labels.detach().cpu().numpy(),
#             )
#         metrics = metric.compute(
#             num_labels=len(dataset_config.ID_TO_LABEL),
#             ignore_index=255,
#             reduce_labels=False,
#         )
#         # print("Loss:", loss.item())
#         print("Mean_iou:", metrics["mean_iou"])
#         # print("Mean accuracy:", metrics["mean_accuracy"])
#         for id, label in dataset_config.ID_TO_LABEL.items():
#             print(f'Mean_iou for {label} {metrics["per_category_iou"][id]}')
#             # print(f'Accuracy for {label} {metrics["per_category_accuracy"][id]}')
#         return metrics["mean_iou"]

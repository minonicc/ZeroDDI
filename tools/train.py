from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import os.path as osp
import copy
from torch.optim import Adam, AdamW
import torch
import time
from .utils import softmax
from tools.logging_ import get_root_logger
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import pickle
import json
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


global history
history = defaultdict(list)


class EvidenceDiagnosticsAccumulator:
    def __init__(self):
        self.values = defaultdict(list)
        self.gate_sum_by_class = None
        self.gate_count_by_class = 0
        self.gate_histogram = torch.zeros(10, dtype=torch.float64)
        self.gate_histogram_by_class = None
        self.gate_value_count = 0
        self.gate_near_zero_count = 0
        self.gate_near_one_count = 0
        self.drug_type_selected = torch.zeros(6, dtype=torch.float64)
        self.drug_type_available = torch.zeros(6, dtype=torch.float64)
        self.pair_type_selected = torch.zeros(36, dtype=torch.float64)
        self.pair_type_available = torch.zeros(36, dtype=torch.float64)
        self.selection_position_histogram = torch.zeros(5, dtype=torch.float64)
        self.selection_position_count = 0

    def update(self, diagnostics):
        if not diagnostics:
            return
        attention = diagnostics.get("pharmacophore_attention")
        if attention is not None:
            probability = attention.detach().float()
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            self.values["attention_entropy"].append(entropy.mean().cpu())
            self.values["attention_top1_mass"].append(
                probability.max(dim=-1).values.mean().cpu()
            )
            self.values["attention_top5_mass"].append(
                probability.topk(min(5, probability.size(-1)), dim=-1).values.sum(dim=-1).mean().cpu()
            )
            self.values["null_token_weight"].append(probability[..., -1].mean().cpu())
        valid_count = diagnostics.get("pharmacophore_valid_pair_count")
        if valid_count is not None:
            self.values["valid_pair_count"].append(
                valid_count.detach().float().mean().cpu()
            )
        selection_indices = diagnostics.get("pharmacophore_selection_indices")
        selection_mask = diagnostics.get("pharmacophore_selection_mask")
        if selection_indices is not None and selection_mask is not None:
            real_mask = selection_mask[..., :selection_indices.size(-1)].detach()
            indices = selection_indices.detach()
            if real_mask.any():
                selected_positions = indices[real_mask].float()
                self.values["selected_original_position_mean"].append(
                    selected_positions.mean().cpu()
                )
                self.values["fixed128_topk_overlap"].append(
                    (selected_positions < 128).float().mean().cpu()
                )
                position_bins = torch.tensor(
                    [64, 128, 256, 512],
                    dtype=selected_positions.dtype,
                    device=selected_positions.device,
                )
                position_bucket = torch.bucketize(
                    selected_positions, position_bins, right=True
                )
                self.selection_position_histogram += torch.bincount(
                    position_bucket.long(), minlength=5
                ).double().cpu()
                self.selection_position_count += selected_positions.numel()
            pair_types = diagnostics.get("pharmacophore_pair_types")
            if pair_types is not None:
                pair_types = pair_types.detach()
                batch_size, num_candidates, _ = selection_indices.shape
                selected_types = torch.gather(
                    pair_types.unsqueeze(1).expand(-1, num_candidates, -1),
                    2,
                    selection_indices,
                )
                available_counts = torch.zeros(
                    batch_size, 36, dtype=torch.long, device=pair_types.device
                )
                available_counts.scatter_add_(
                    1,
                    pair_types.clamp_min(0),
                    (pair_types >= 0).long(),
                )
                selected_counts = torch.zeros(
                    batch_size,
                    num_candidates,
                    36,
                    dtype=torch.long,
                    device=pair_types.device,
                )
                selected_counts.scatter_add_(
                    2,
                    selected_types.clamp_min(0),
                    real_mask.long(),
                )
                available_presence = available_counts.gt(0).unsqueeze(1)
                selected_presence = selected_counts.gt(0)
                coverage = selected_presence.sum(dim=-1).float() / available_presence.sum(
                    dim=-1
                ).clamp_min(1)
                self.values["pair_topk_type_coverage"].append(coverage.mean().cpu())
                self.pair_type_available += (
                    available_counts.sum(dim=0).double().cpu() * num_candidates
                )
                self.pair_type_selected += selected_counts.sum((0, 1)).double().cpu()
        gate = diagnostics.get("pharmacophore_gate")
        if gate is not None:
            gate = gate.detach().float().squeeze(-1)
            self.values["gate_mean"].append(gate.mean().cpu())
            self.values["gate_std"].append(gate.std(unbiased=False).cpu())
            gate_cpu = gate.cpu()
            self.gate_histogram += torch.histc(
                gate_cpu, bins=10, min=0.0, max=1.0
            ).double()
            gate_bucket = (gate_cpu * 10).long().clamp_(0, 9)
            class_offsets = torch.arange(gate_cpu.size(1)).view(1, -1) * 10
            class_histogram = torch.bincount(
                (gate_bucket + class_offsets).reshape(-1),
                minlength=gate_cpu.size(1) * 10,
            ).reshape(gate_cpu.size(1), 10).double()
            self.gate_histogram_by_class = (
                class_histogram
                if self.gate_histogram_by_class is None
                else self.gate_histogram_by_class + class_histogram
            )
            self.gate_value_count += gate_cpu.numel()
            self.gate_near_zero_count += int((gate_cpu <= 0.05).sum())
            self.gate_near_one_count += int((gate_cpu >= 0.95).sum())
            class_sum = gate.sum(dim=0).cpu()
            self.gate_sum_by_class = (
                class_sum if self.gate_sum_by_class is None
                else self.gate_sum_by_class + class_sum
            )
            self.gate_count_by_class += gate.size(0)
        drug_selection = diagnostics.get("pharmacophore_drug_selection")
        if drug_selection is not None:
            self.values["valid_pair_count"].append(
                drug_selection["pair_mask"][..., :-1]
                .detach()
                .float()
                .sum(dim=-1)
                .mean()
                .cpu()
            )
            num_classes = drug_selection["selected_types_a"].size(1)
            for side in ("a", "b"):
                source_types = drug_selection[f"source_types_{side}"].detach().cpu()
                source_mask = drug_selection[f"source_mask_{side}"].detach().cpu()
                source_count = torch.bincount(
                    source_types[source_mask], minlength=6
                ).double()
                self.drug_type_available += source_count * num_classes
                selected_types = drug_selection[f"selected_types_{side}"].detach().cpu()
                selected_valid = drug_selection[f"valid_{side}"].detach().cpu()
                self.drug_type_selected += torch.bincount(
                    selected_types[selected_valid], minlength=6
                ).double()

    def summarize(self):
        summary = {
            name: float(torch.stack(values).mean())
            for name, values in self.values.items()
            if values
        }
        if self.gate_sum_by_class is not None and self.gate_count_by_class:
            summary["gate_mean_by_class"] = (
                self.gate_sum_by_class / self.gate_count_by_class
            ).tolist()
        if self.gate_value_count:
            summary["gate_histogram_10bin"] = (
                self.gate_histogram / self.gate_value_count
            ).tolist()
            summary["gate_histogram_10bin_by_class"] = (
                self.gate_histogram_by_class / self.gate_count_by_class
            ).tolist()
            summary["gate_fraction_le_0_05"] = (
                self.gate_near_zero_count / self.gate_value_count
            )
            summary["gate_fraction_ge_0_95"] = (
                self.gate_near_one_count / self.gate_value_count
            )
        if self.drug_type_available.sum() > 0:
            summary["drug_topk_type_retention"] = (
                self.drug_type_selected / self.drug_type_available.clamp_min(1)
            ).tolist()
        if self.pair_type_available.sum() > 0:
            summary["pair_topk_type_retention"] = (
                self.pair_type_selected / self.pair_type_available.clamp_min(1)
            ).tolist()
        if self.selection_position_count:
            summary["selected_position_histogram_64_128_256_512"] = (
                self.selection_position_histogram / self.selection_position_count
            ).tolist()
        return summary


def train_model(model, datasets, cfg):
    logger = get_root_logger(log_level=cfg.log_level)
    train_sampler = RandomSampler(datasets[0]) #datasets[0] is train dataset
    logger.info("The number of train instances: {}".format(len(train_sampler)))


    use_collate_fn = False
    if use_collate_fn:
        train_dataloader = DataLoader(datasets[0], sampler=train_sampler,
                                      batch_size=cfg.train_batch_size,
                                      collate_fn=datasets[0].collate_fn)
    else:
        train_dataloader = DataLoader(datasets[0], sampler=train_sampler,
                                      batch_size=cfg.train_batch_size)


    no_decay = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.001}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=cfg.learning_rate)
    torch.autograd.set_detect_anomaly(cfg.get('detect_anomaly', True))

    zsl_best_model = 0
    gzsl_best_model = 0
    zsl_best_epoch = 0
    gzsl_best_epoch = 0
    best_H = 0
    best_per_acc = 0
    seen_best_model = 0
    seen_best_epoch = 0
    best_seen_acc = 0
    best_seen_metrics = None
    eval_modes = cfg.get('eval_modes', ['zsl', 'gzsl'])
    eval_interval = cfg.get('eval_interval', 1)

    for epoch in range(cfg.num_epochs):
        epoch_diagnostics = EvidenceDiagnosticsAccumulator()
        batch_step = 0
        batch_loss = 0
        #train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            max_train_steps = cfg.get("max_train_steps_per_epoch", None)
            if max_train_steps is not None and step >= max_train_steps:
                break
            t1 = time.time()
            model.train()
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, step {step + 1}: {loss.item()}"
                )
            if len(outputs) > 8:
                epoch_diagnostics.update(outputs[8])

            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_step += 1
            log_interval = cfg.get("log_config", {}).get("interval", 50)
            if log_interval and (step + 1) % log_interval == 0:
                logger.info(
                    "epoch:%d step:%d/%d running_train_loss:%f",
                    epoch + 1,
                    step + 1,
                    len(train_dataloader),
                    batch_loss / batch_step,
                )
            
        if (epoch + 1) % eval_interval == 0:

            logger.info(f"epoch is {epoch} || Train batch_loss is {batch_loss / batch_step} \n")
            history['train_loss'].append(batch_loss / batch_step)
            if 'zsl' in eval_modes:
                per_acc = evaluate(model, datasets[1], logger, cfg, "zsl", "test")  # zsl_val_dataset
                if per_acc > best_per_acc:
                    best_per_acc = per_acc
                    zsl_best_epoch = epoch
                    zsl_best_model = copy.deepcopy(model.state_dict())

            if 'gzsl' in eval_modes:
                H = evaluate(model, datasets[2], logger, cfg, "gzsl", "test")  # gzsl_val_dataset
                if H > best_H:
                    best_H = H
                    gzsl_best_epoch = epoch
                    gzsl_best_model = copy.deepcopy(model.state_dict())

            if 'seen' in eval_modes:
                seen_metrics = evaluate(
                    model, datasets[1], logger, cfg, "seen", "test", return_metrics=True
                )
                seen_acc = seen_metrics[cfg.get("selection_metric", "Macro-F1")]
                if is_better_checkpoint(seen_metrics, best_seen_metrics):
                    best_seen_acc = seen_acc
                    best_seen_metrics = seen_metrics
                    seen_best_epoch = epoch
                    seen_best_model = copy.deepcopy(model.state_dict())
                    # Persist every validation improvement so a long run keeps
                    # its best checkpoint even if it ends before the next
                    # periodic snapshot or the normal final save.
                    torch.save(seen_best_model, cfg.model_parameter_best)
                    logger.info(
                        "Saved improved validation checkpoint at epoch %d to %s",
                        epoch + 1,
                        cfg.model_parameter_best,
                    )
            alpha = getattr(model.Leftmodel, "fixed_substructure_alpha", None)
            if alpha is not None:
                alpha_value = float(alpha.detach().cpu())
                history['fixed_substructure_alpha'].append(alpha_value)
                logger.info("fixed_substructure_alpha:%f", alpha_value)
            diagnostic_summary = epoch_diagnostics.summarize()
            if diagnostic_summary:
                diagnostic_summary["epoch"] = epoch + 1
                diagnostic_path = osp.join(
                    cfg.work_dir, f"diagnostics_seed{cfg.seednumber}.jsonl"
                )
                with open(diagnostic_path, "a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(diagnostic_summary) + "\n")
                logger.info("evidence_diagnostics:%s", diagnostic_summary)
            # print("time",time.time()-t1)
        if (epoch + 1) % 20 == 0:
            #if torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                    f'model_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))
            if zsl_best_model != 0:
                torch.save(zsl_best_model, osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                    f'zsl_model_best_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))
            if gzsl_best_model != 0:
                torch.save(gzsl_best_model, osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                     f'gzsl_model_best_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))
            if seen_best_model != 0:
                torch.save(seen_best_model, osp.join(osp.join(cfg.work_dir, 'model_parameter'),
                                                     f'seen_model_best_epoch{epoch + 1}_seed{cfg.seednumber}.pkl'))

    logger.info(f"The gzsl best epoch is {gzsl_best_epoch + 1}")
    logger.info(f"The zsl best epoch is {zsl_best_epoch + 1}")
    logger.info(f"The seen best epoch is {seen_best_epoch + 1}")
    if seen_best_model != 0:
        torch.save(seen_best_model, cfg.model_parameter_best)
        logger.info(
            "Saved validation-selected seen checkpoint to %s with metrics %s",
            cfg.model_parameter_best,
            best_seen_metrics,
        )
    # save history



def is_better_checkpoint(metrics, best_metrics, macro_f1_tolerance=0.001):
    """Compare validation checkpoints using Macro-F1, then Kappa and ACC."""
    if best_metrics is None:
        return True
    macro_delta = metrics["Macro-F1"] - best_metrics["Macro-F1"]
    if abs(macro_delta) >= macro_f1_tolerance:
        return macro_delta > 0
    if metrics["Kappa"] != best_metrics["Kappa"]:
        return metrics["Kappa"] > best_metrics["Kappa"]
    return metrics["ACC"] > best_metrics["ACC"]


def evaluate(
    model,
    dataset,
    logger,
    cfg,
    zsl,
    aaa="test",
    visualize_acc=False,
    return_metrics=False,
):

    eval_sampler = SequentialSampler(dataset)
    use_collate_fn = False
    evel_data_len = len(dataset)
    if use_collate_fn:
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                     batch_size=64, collate_fn=dataset.collate_fn,drop_last=False)
    else:
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                     batch_size=64,drop_last=False)
      
    mode = dataset.mode
    logger.info(
        f"The number of {mode} instances: {len(eval_sampler)}, the class has: {len(dataset.current_dataset_eventid_uni)}")
    embid2eventid = dataset.embid2eventid

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    gt_emb_ids = None
    instances = None
    prototypes = None
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch)
    
        loss, Logits_all, gt_id, instance, prototype = outputs[:5]  # (cost, logits, emb_ids, Matmul_gnn_W, right_output_all)
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = Logits_all.detach().cpu().numpy()
            gt_emb_ids = gt_id.detach().cpu().numpy()
            instances = instance.detach().cpu().numpy()
            prototypes = prototype.detach().cpu().numpy()
        else:
            preds = np.append(preds, Logits_all.detach().cpu().numpy(), axis=0)
            gt_emb_ids = np.append(gt_emb_ids, gt_id.detach().cpu().numpy(), axis=0)
            instances = np.append(instances, instance.detach().cpu().numpy(), axis=0)
            prototypes = np.append(prototypes, prototype.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if zsl == "zsl" or zsl == "seen":
        Val_Evaluation, per_class_top_1_acc, acc_per_class_list, true_label_count = zsl_accuracy(preds, gt_emb_ids)


        logger.info(f"*****Begin to {mode}, eval loss: {eval_loss}****** ")
        logger.info(
            "per_class_top@1_acc:%f , top@1 acc:%f, top@2 acc:%f, top@3 acc:%f, top@5 acc:%f" % (
                per_class_top_1_acc,
                Val_Evaluation["Accuracy"],
                Val_Evaluation["Top2Acc"],
                Val_Evaluation["Top3Acc"],
                Val_Evaluation["Top5Acc"]))
        cls_metrics = classification_metrics(preds, gt_emb_ids)
        log_classification_metrics(logger, cls_metrics)
        if visualize_acc:
            save_classification_details(
                preds,
                gt_emb_ids,
                cfg,
                mode,
                embid2eventid=embid2eventid,
            )
        logger.info("************************\n")

        if return_metrics:
            return cls_metrics
        selection_metric = cfg.get("selection_metric", "per_class_top@1_acc")
        if selection_metric == "per_class_top@1_acc":
            return per_class_top_1_acc
        if selection_metric not in cls_metrics:
            raise KeyError(
                f"Unsupported selection_metric: {selection_metric}. "
                f"Available metrics: {list(cls_metrics.keys()) + ['per_class_top@1_acc']}"
            )
        return cls_metrics[selection_metric]

    elif zsl == "gzsl":
        # seen_labels, embedid2eventid, H = model.evaluate(mode, preds, gt_ids, instances, prototypes,logger,zsl)
        Val_Evaluation, top1_acc_seen, top1_acc_unseen,top1H, per_class_top_1_acc, acc_seen, \
        acc_unseen, H, two_classify_seenacc, \
        two_classify_unseenacc, bi_acc, acc_per_class_list_seen, true_label_count_seen, \
        acc_per_class_list_unseen, true_label_count_unseen = gzsl_accuracy(preds, gt_emb_ids, embid2eventid,
                                                                           cfg.model.seen_labels)



        logger.info(f"*****Begin to {mode}, eval loss: {eval_loss}****** ")
        logger.info(
            "per_class_top@1_acc:%f , top@1 acc:%f, top@2 acc:%f, top@3 acc:%f, "
            "top@5 acc:%f, top1_acc_seen:%f, top1_acc_unseen: %f, top1_H: %f, "
            "per_seenacc:%f, per_unseenacc:%f, H:%f, two_classify_seenacc:%f, "
            "two_classify_unseenacc:%f, bi_acc:%f" % (
                per_class_top_1_acc,
                Val_Evaluation["Accuracy"],
                Val_Evaluation["Top2Acc"],
                Val_Evaluation["Top3Acc"],
                Val_Evaluation["Top5Acc"],
                top1_acc_seen,
                top1_acc_unseen,
                top1H,
                acc_seen,
                acc_unseen,
                H,
                two_classify_seenacc,
                two_classify_unseenacc,
                bi_acc))
        cls_metrics = classification_metrics(preds, gt_emb_ids)
        log_classification_metrics(logger, cls_metrics)
        logger.info("************************\n")

        return H




def classification_metrics(logits, ids):
    probabilities = softmax(logits, axis=1)
    preds = np.argmax(logits, axis=1)
    ids = np.asarray(ids)

    metrics = {
        "ACC": accuracy_score(ids, preds),
        "Kappa": cohen_kappa_score(ids, preds),
        "Macro-F1": f1_score(ids, preds, average="macro", zero_division=0),
        "Weighted-F1": f1_score(ids, preds, average="weighted", zero_division=0),
        "Macro-Precision": precision_score(ids, preds, average="macro", zero_division=0),
        "Macro-Recall": recall_score(ids, preds, average="macro", zero_division=0),
    }

    class_ids = np.arange(probabilities.shape[1])
    y_true = (ids[:, None] == class_ids[None, :]).astype(int)
    present = y_true.sum(axis=0) > 0
    if present.any():
        metrics["PR-AUC-macro"] = average_precision_score(
            y_true[:, present],
            probabilities[:, present],
            average="macro",
        )
        metrics["PR-AUC-micro"] = average_precision_score(
            y_true[:, present],
            probabilities[:, present],
            average="micro",
        )
    else:
        metrics["PR-AUC-macro"] = float("nan")
        metrics["PR-AUC-micro"] = float("nan")

    return metrics


def log_classification_metrics(logger, metrics):
    logger.info(
        "ACC:%f, Kappa:%f, Macro-F1:%f, Weighted-F1:%f, "
        "Macro-Precision:%f, Macro-Recall:%f, PR-AUC-macro:%f, PR-AUC-micro:%f"
        % (
            metrics["ACC"],
            metrics["Kappa"],
            metrics["Macro-F1"],
            metrics["Weighted-F1"],
            metrics["Macro-Precision"],
            metrics["Macro-Recall"],
            metrics["PR-AUC-macro"],
            metrics["PR-AUC-micro"],
        )
    )


def save_classification_details(
    logits, ids, cfg, mode, embid2eventid=None
):
    predictions = np.argmax(logits, axis=1)
    labels = np.arange(logits.shape[1])
    precision, recall, f1, support = precision_recall_fscore_support(
        ids, predictions, labels=labels, zero_division=0
    )
    details_data = {"class_id": labels}
    if embid2eventid is not None:
        details_data["event_id"] = [
            embid2eventid.get(int(label), int(label))
            if hasattr(embid2eventid, "get")
            else embid2eventid[int(label)]
            for label in labels
        ]
    details_data.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    details = pd.DataFrame(details_data)
    details.to_csv(
        osp.join(cfg.work_dir, f"{mode}_class_metrics_seed{cfg.seednumber}.csv"),
        index=False,
    )
    np.save(
        osp.join(cfg.work_dir, f"{mode}_confusion_matrix_seed{cfg.seednumber}.npy"),
        confusion_matrix(ids, predictions, labels=labels),
    )


def zsl_accuracy(logits, ids):
    probabilities = softmax(logits, axis=1)

    get_top_n = (-probabilities).argsort()[:, 0:10]
    preds = np.argmax(logits, axis=1)

    Val_Evaluation = GetAccuracy(preds, get_top_n, ids)
    per_class_top_1_acc, acc_per_class_list, true_label_count = average_per_class_top_1_acc(preds, ids)
    return Val_Evaluation, per_class_top_1_acc, acc_per_class_list, true_label_count


def gzsl_accuracy(logits, ids, embedid2eventid, seen_labels):
    probabilities = softmax(logits, axis=1)

    get_top_n = (-probabilities).argsort()[:, 0:10]
    preds = np.argmax(logits, axis=1)

    Val_Evaluation = GetAccuracy(preds, get_top_n, ids)
    per_class_top_1_acc, _, _ = average_per_class_top_1_acc(preds, ids)

    # change
    true_eventid_list = []
    for i in range(ids.shape[0]):
        true_eventid_list.append(embedid2eventid[ids[i]])

    pred_top1_id = list(np.squeeze((-probabilities).argsort()[:, 0]))
    pred_eventid_list = []
    for i in pred_top1_id:
        pred_eventid_list.append(embedid2eventid[i])
    true_eventid = np.array(true_eventid_list)
    pred_eventid = np.array(pred_eventid_list)

    # mask the train triplet
    mask_seen = [index for index, tri in enumerate(true_eventid) if tri in seen_labels]
    mask_unseen = [index for index, tri in enumerate(true_eventid) if tri not in seen_labels]

    pred_seen = pred_eventid[mask_seen]
    true_seen = true_eventid[mask_seen]
    pred_unseen = pred_eventid[mask_unseen]
    true_unseen = true_eventid[mask_unseen]

    per_acc_seen, acc_per_class_list_seen, true_label_count_seen = average_per_class_top_1_acc(pred_seen, true_seen)
    per_acc_unseen, acc_per_class_list_unseen, true_label_count_unseen = average_per_class_top_1_acc(pred_unseen,
                                                                                                     true_unseen)
    match_seen = np.equal(pred_seen, true_seen)
    match_unseen = np.equal(pred_unseen, true_unseen)
    top1_acc_seen = np.sum(match_seen) / len(mask_seen)
    top1_acc_unseen = np.sum(match_unseen) / len(mask_unseen)
    top1H = 2 * (top1_acc_seen * top1_acc_unseen) / (top1_acc_seen + top1_acc_unseen)

    H = 2 * (per_acc_seen * per_acc_unseen) / (per_acc_seen + per_acc_unseen)

    # Binary classification
    true_seen_eventid = set(list(true_seen))
    true_unseen_eventid = set(list(true_unseen))
    gt_twoclassify_label = []
    pred_twoclassify_label = []
    for i in true_eventid_list:
        if i in true_seen_eventid:
            gt_twoclassify_label.append(
                1)  # if true eventid belong to seen class, 1; elif true eventid belong to unseen class, 0
        elif i in true_unseen_eventid:
            gt_twoclassify_label.append(0)
    assert len(gt_twoclassify_label) == len(true_eventid_list)

    for i in pred_eventid_list:
        if i in true_seen_eventid:
            pred_twoclassify_label.append(
                1)  # if true eventid belong to seen class, 1; elif true eventid belong to unseen class, 0
        elif i in true_unseen_eventid:
            pred_twoclassify_label.append(0)
        else:
            pred_twoclassify_label.append(9)
    assert len(pred_twoclassify_label) == len(pred_eventid_list)

    two_classify_matchseen = np.equal(np.array(gt_twoclassify_label)[mask_seen],
                                      np.array(pred_twoclassify_label)[mask_seen])
    two_classify_seenacc = np.sum(two_classify_matchseen != 0) / len(mask_seen)
    two_classify_matchunseen = np.equal(np.array(gt_twoclassify_label)[mask_unseen],
                                        np.array(pred_twoclassify_label)[mask_unseen])

    two_classify_unseenacc = np.sum(two_classify_matchunseen != 0) / len(mask_unseen)
    bi_acc = np.sum(np.equal(np.array(gt_twoclassify_label),
                             np.array(pred_twoclassify_label)) != 0) / len(gt_twoclassify_label)

    return Val_Evaluation, top1_acc_seen, top1_acc_unseen, top1H, per_class_top_1_acc, per_acc_seen, per_acc_unseen, H, \
           two_classify_seenacc, two_classify_unseenacc, bi_acc, acc_per_class_list_seen, true_label_count_seen, \
           acc_per_class_list_unseen, true_label_count_unseen


def GetAccuracy(Y_Pred, top_10_classes, Y_True):
    """ Returns Accuracy when multi-label are provided for each instance. It will be counted true if predicted y is among the true labels
    Args:
        Y_Pred (int array): the predicted labels
        Probabilities (float [][] array): the probabilities predicted for each class for each instance
        Y_True (int[] array): the true labels, for each instance it should be a list
    """

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    count_true = 0
    count_true_2 = 0
    count_true_3 = 0
    count_true_5 = 0
    count_true_10 = 0
    for i in range(len(Y_Pred)):
        if Y_Pred[i] == Y_True[i]:
            count_true += 1
        if len(intersection(top_10_classes[i], [Y_True[i]])) > 0:
            count_true_10 += 1
        if len(intersection(top_10_classes[i][:5], [Y_True[i]])) > 0:
            count_true_5 += 1
        if len(intersection(top_10_classes[i][:3], [Y_True[i]])) > 0:
            count_true_3 += 1
        if len(intersection(top_10_classes[i][:2], [Y_True[i]])) > 0:
            count_true_2 += 1

    Evaluations = {"Accuracy": (float(count_true) / len(Y_Pred)),
                   "Top2Acc": (float(count_true_2) / len(Y_Pred)),
                   "Top3Acc": (float(count_true_3) / len(Y_Pred)),
                   "Top5Acc": (float(count_true_5) / len(Y_Pred))}
    return Evaluations


def average_per_class_top_1_acc(preds, batch_true_label_):
    """
    Zero-Shot Learning - The Good, the Bad and the Ugly
    """
    split_class = defaultdict(list)
    true_label_count = defaultdict(list)

    for i, label in enumerate(batch_true_label_):
        split_class[label].append(preds[i])
        true_label_count[label].append(1)

    acc_per_class_list = []
    ids = list(split_class.keys())

    for key, values in split_class.items():
        length = len(values)
        count = 0
        for v in values:
            if v == key:
                count += 1
        acc_per_class = count / length
        acc_per_class_list.append(acc_per_class)
    per_class_top_1_acc = np.mean(np.array(acc_per_class_list))
    return per_class_top_1_acc, acc_per_class_list, true_label_count




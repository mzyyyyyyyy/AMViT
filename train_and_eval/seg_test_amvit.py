import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import numpy as np
from utils.lr_scheduler import build_scheduler
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, load_from_checkpoint
from data import get_dataloaders
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from data import get_loss_data_input

from tqdm import tqdm

def test(net, testloader, loss_fn, config, device, loss_input_fn):
    num_classes = config['MODEL']['num_classes'] - len(config['MODEL']['ignore_background'])
    predicted_all = []
    labels_all = []
    losses_all = []
    net.eval()
    with torch.no_grad():
        for step, sample in tqdm(enumerate(testloader)):
            logits = net(sample['inputs'].to(device))
            logits = logits.permute(0, 2, 3, 1)
            _, predicted = torch.max(logits.data, -1)
            # predicted.shape = (24, 24, 24)
            ground_truth = loss_input_fn(sample, device)
            loss = loss_fn['all'](logits, ground_truth)
            target, mask = ground_truth
            if mask is not None:
                predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
            else:
                predicted_all.append(predicted.view(-1).cpu().numpy())
                labels_all.append(target.view(-1).cpu().numpy())
            losses_all.append(loss.view(-1).cpu().detach().numpy())

            # 计算并打印每个 batch 的评估结果
            predicted_classes = np.concatenate(predicted_all)
            target_classes = np.concatenate(labels_all)
            losses = np.concatenate(losses_all)

            eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                    n_classes=config['MODEL']['num_classes'], cm=False, unk_masks=None)

            micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
            macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']

            # print(f"Batch {step + 1}:")
            # print(f"Loss: {losses.mean():.7f}")
            # print(f"Micro IOU: {micro_IOU:.4f}, Macro IOU: {macro_IOU:.4f}")
            # print(f"Micro Accuracy: {micro_acc:.4f}, Macro Accuracy: {macro_acc:.4f}")
            # print(f"Micro Precision: {micro_precision:.4f}, Macro Precision: {macro_precision:.4f}")
            # print(f"Micro Recall: {micro_recall:.4f}, Macro Recall: {macro_recall:.4f}")
            # print(f"Micro F1: {micro_F1:.4f}, Macro F1: {macro_F1:.4f}")
            # print("-" * 100)




        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, cm=True, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                         help='gpu ids to use')

    args = parser.parse_args()
    config_file = args.config
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]

    device = get_device(device_ids, allow_cpu=True)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False, device=config['MODEL']['device'])

    net.to(device)

    loss_input_fn = get_loss_data_input(config)
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    test_metrics = test(net, dataloaders['test'], loss_fn, config, device, loss_input_fn)
    print(test_metrics)

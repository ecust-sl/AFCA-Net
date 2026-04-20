'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''
from sklearn.metrics import roc_auc_score, accuracy_score
from setting import parse_opts
import sys

sys.path.append("/home/shilei/project/MedicalNet")
# from tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
import numpy as np
import swanlab
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
# 输入输出路径配置
input_txt = "/extra/shilei/tongji_newdata/all_data.txt"  # 原始数据文件
train_txt = "/extra/shilei/tongji_newdata/train_new_line.txt"  # 输出训练集
val_txt = "/extra/shilei/tongji_newdata/test_new_line.txt"  # 输出验证集
test_size = 0.2  # 验证集比例(1/5=0.2)
random_seed = 42  # 随机种子保证可重复性


def compute_multiclass_specificity(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)
    specificity = []
    for c in classes:
        # 将多分类问题转换为二分类问题
        y_true_binary = (y_true == c)
        y_pred_binary = (y_pred == c)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        specificity.append(tn / (tn + fp))

    if average == 'macro':
        return np.mean(specificity)
    elif average == 'weighted':
        weights = np.bincount(y_true)[classes] / len(y_true)
        return np.sum(specificity * weights)
    else:
        return specificity


# 在代码开头添加自定义指标计算函数
import numpy as np


def train(data_loader, test_loader, val_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets,
          num_KF):
    # settings
    acc_max, f1_max, recall_max, auc_max = 0, 0, 0, 0
    swanlab.init(
        project="Med3D-KF-5",
        name=f"0621_fold_{num_KF}_2_3_loss_fusion_add",
        config={
            "learning_rate": scheduler.get_last_lr()[0],
            "epochs": total_epochs,
            "batch_size": data_loader.batch_size,
            "fold": num_KF
        }
    )
    best_metrics = {
        'val_auc': 0,
        'val_f1': 0,
        'val_acc': 0,
        'val_recall': 0,
        'val_auc_3': 0,
        'val_f1_3': 0,
        'val_acc_3': 0,
        'val_recall_3': 0,
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() and not sets.no_cuda else "cpu")
    model.to(device)
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # loss_cls = nn.CrossEntropyLoss()
    # loss_cls = nn.BCELoss()
    # loss_cls= nn.BCEWithLogitsLoss()
    loss_cls_2 = torch.nn.BCEWithLogitsLoss().cuda()
    loss_cls_3 = torch.nn.CrossEntropyLoss().cuda()

    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        # loss_seg = loss_seg.cuda()
        loss_cls_2 = loss_cls_2.to(device)
        loss_cls_3 = loss_cls_3.to(device)

    train_time_sp = time.time()
    for epoch in range(total_epochs):
        model.to(device)
        model.train()

        log.info('Start epoch {}'.format(epoch))

        log.info('lr = {}'.format(scheduler.get_lr()))
        per_epoch_loss = 0
        num_correct = 0
        all_probs = []
        all_probs_3 = []
        all_labels = []
        all_labels_3 = []

        val_num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, flair, label_masks, tss_label, tss_label_3, _,texts_dwi, texts_flair = batch_data
            # print(f"dwi-text: {(texts_dwi)}, flair-text: {(texts_flair)}")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model_bert = BertModel.from_pretrained('bert-base-uncased')
            inputs_dwi = tokenizer(texts_dwi, padding=True, truncation=True, return_tensors='pt')
            inputs_flair = tokenizer(texts_flair, padding=True, truncation=True, return_tensors='pt')
            # print(f'input_ids shape: {inputs["input_ids"].shape}')
            with torch.no_grad():
                outputs1 = model_bert(**inputs_dwi)
                outputs2 = model_bert(**inputs_flair)
            dwi_text_features = outputs1.last_hidden_state.to(device)
            flair_text_features = outputs2.last_hidden_state.to(device)
            # print(label_masks.shape)
            # tss_label = tss_label.float().view(-1, 1)
            if not sets.no_cuda:
                volumes = volumes.to(device)
                flair = flair.to(device)
                tss_label = tss_label.to(device).float()
                tss_label_3 = tss_label_3.to(device).long()

            # out_masks = model(volumes)
            # fused_feat = torch.cat([volumes, flair], dim=1)
            # fused_feat = volumes
            logits_2, logits_3 = model(volumes, flair, dwi_text_features, flair_text_features)
            logits_2 = logits_2.reshape(-1)
            probabilities = torch.softmax(logits_3, dim=1)  # 计算概率

            # logits.to(device)
            # logits = logits.reshape(-1)
            prob_out = nn.Sigmoid()(logits_2)
            # pro_list = prob_out.detach().cpu().numpy()

            # print(f'device1:{logits.device}- devicd2:{tss_label.device}')
            loss_cls_value_2 = loss_cls_2(logits_2, tss_label)
            loss_cls_value_3 = loss_cls_3(logits_3, tss_label_3)

            all_probs.append(prob_out.detach().cpu().numpy())  # 分离梯度并转CPU
            all_labels.append(tss_label.detach().cpu().numpy())  # 分离梯度并转CPU
            all_probs_3.append(probabilities.detach().cpu().numpy())  # 保存概率矩阵
            all_labels_3.append(tss_label_3.detach().cpu().numpy())
            # preds = (prob_out > 0.5).float()  # 直接生成预测结果
            # num_correct += (preds == tss_label).sum().item()  # 直接统计正确数
            # loss = loss_value_seg
            loss = loss_cls_value_2 + loss_cls_value_3
            per_epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # preds = logits.argmax(dim=1)

            # labels = tss_label.cpu().numpy()

            # all_preds.extend(preds)
            # all_labels.extend(labels)
            # num_correct += torch.eq(preds, tss_label).sum().float().item()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_2 = {:.3f}, loss_3 = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_cls_value_2.item(), loss_cls_value_3.item(), avg_batch_time))
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_probs_3 = np.concatenate(all_probs_3)
        all_labels_3 = np.concatenate(all_labels_3)
        # print('probs_',all_probs_3)

        # 根据阈值0.5生成预测类别
        pred_labels = (all_probs > 0.5).astype(int)
        pred_labels_3 = np.argmax(all_probs_3, axis=1)
        # acc = accuracy_score(all_labels, pred_labels)  # 官方Acc计算
        # manual_results = manual_metrics(all_labels, pred_labels)
        try:
            from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
            lib_acc = accuracy_score(all_labels, pred_labels)
            lib_recall = recall_score(all_labels, pred_labels)
            lib_f1 = f1_score(all_labels, pred_labels)
            lib_auc = roc_auc_score(all_labels, all_probs)
            lib_acc_3 = accuracy_score(all_labels_3, pred_labels_3)
            lib_recall_3 = recall_score(all_labels_3, pred_labels_3,average='macro')
            lib_f1_3 = f1_score(all_labels_3, pred_labels_3, average='macro')
            lib_auc_3 = roc_auc_score(all_labels_3, all_probs_3, multi_class='ovr', average='macro')
        except ImportError:
            lib_acc = lib_recall = lib_f1 = lib_auc = -1  # 标记库不可用
            lib_acc_3 = lib_recall_3 = lib_f1_3 = lib_auc_3 = -1  # 标记库不可用
        log.info(
            "Train Epoch: {}\t Loss: {:.6f}\t "
            "Binary - Acc:{:.4f} Recall:{:.4f} F1:{:.4f} AUC:{:.4f}\t "
            "3-Class - Acc:{:.4f} Recall:{:.4f} F1:{:.4f} AUC:{:.4f}".format(
                epoch,
                per_epoch_loss,
                lib_acc, lib_recall, lib_f1, lib_auc,
                lib_acc_3, lib_recall_3, lib_f1_3, lib_auc_3
            )
        )
        train_metrics = {
            'epoch': epoch,
            'fold': num_KF,
            'train/loss': per_epoch_loss / len(data_loader),
            'train/acc': lib_acc,
            'train/f1': lib_f1,
            'train/recall': lib_recall,
            'train/auc': lib_auc,
            'train/acc_3': lib_acc_3,
            'train/f1_3': lib_f1_3,
            'train/recall_3': lib_recall_3,
            'train/auc_3': lib_auc_3,
        }

        # print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.4f}\t AUC: {:.4f}".format(
        #     epoch,
        #     per_epoch_loss,
        #     acc,
        #     auc
        # ))
        model.eval()
        all_val_probs = []
        all_val_probs_3 = []
        all_val_labels = []
        all_val_labels_3 = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(test_loader):
                # getting data batch
                # batch_id_sp = epoch * batches_per_epoch
                val_volumes, val_flair, val_label_masks, val_tss_label, val_tss_label_3, _, texts_dwi, texts_flair = batch_data
                # print(f"dwi-text-shape: {len(texts_dwi)}, flair-text-shape: {len(texts_flair)}")
                # print(f"dwi-text: {(texts_dwi)}, flair-text: {(texts_flair)}")
                # for i, x in enumerate(texts_flair):
                #     print(f"[{i}] → type={type(x)}, is str? {isinstance(x, str)}, repr={x[:50]!r}…")
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model_bert = BertModel.from_pretrained('bert-base-uncased')
                inputs_dwi = tokenizer(texts_dwi, padding=True, truncation=True, return_tensors='pt')
                inputs_flair = tokenizer(texts_flair, padding=True, truncation=True, return_tensors='pt')
                # print(f'input_ids shape: {inputs["input_ids"].shape}')
                with torch.no_grad():
                    test_outputs1 = model_bert(**inputs_dwi)
                    test_outputs2 = model_bert(**inputs_flair)
                test_dwi_text_features = test_outputs1.last_hidden_state.to(device)
                test_flair_text_features = test_outputs2.last_hidden_state.to(device)
                # print(f'val-shape:{val_text_features.shape}')shape
                # tss_label = tss_label.float().view(-1, 1)
                if not sets.no_cuda:
                    val_volumes = val_volumes.cuda()
                    val_flair = val_flair.cuda()
                    val_tss_label = val_tss_label.cuda()
                    val_tss_label_3 = val_tss_label_3.cuda()
                    # Forward pass
                # val_fused_feat = torch.cat([val_volumes, val_flair], dim=1)
                logits_2, logits_3 = model(val_volumes, val_flair, test_dwi_text_features, test_flair_text_features)
                logits_2 = logits_2.reshape(val_tss_label.shape[0])  # 直接使用张量形状
                prob_out = nn.Sigmoid()(logits_2)
                probabilities = torch.softmax(logits_3, dim=1)  # 计算概率

                # 收集结果（自动处理设备）
                all_val_probs.append(prob_out.detach().cpu())
                all_val_probs_3.append(probabilities.detach().cpu())
                all_val_labels.append(val_tss_label.detach().cpu())
                all_val_labels_3.append(val_tss_label_3.detach().cpu())

            # 合并结果
            val_probs = torch.cat(all_val_probs).numpy()
            val_probs_3 = torch.cat(all_val_probs_3).numpy()
            val_labels = torch.cat(all_val_labels).numpy()
            val_labels_3 = torch.cat(all_val_labels_3).numpy()

            # 计算指标
            val_preds = (val_probs > 0.5).astype(int)
            val_preds_3 = np.argmax(val_probs_3, axis=1)
            # val_acc = accuracy_score(val_labels, val_preds)

            # manual_results_val = manual_metrics(val_labels, val_preds)
            try:
                from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
                lib_acc_val = accuracy_score(val_labels, val_preds)
                lib_recall_val = recall_score(val_labels, val_preds)
                lib_f1_val = f1_score(val_labels, val_preds)
                lib_auc_val = roc_auc_score(val_labels, val_probs)
                tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()
                lib_specificity_val = tn / (tn + fp)  # 特异性计算
                lib_acc_val_3 = accuracy_score(val_labels_3, val_preds_3)
                lib_recall_val_3 = recall_score(val_labels_3, val_preds_3, average='macro')
                lib_f1_val_3 = f1_score(val_labels_3, val_preds_3, average='macro')
                lib_auc_val_3 = roc_auc_score(val_labels_3, val_probs_3,multi_class='ovr', average='macro')
                lib_specificity_val_3 = compute_multiclass_specificity(
                    val_labels_3,
                    val_preds_3,
                    average='macro'  # 保持与之前指标一致的宏平均
                )
            except ImportError:
                lib_acc_val = lib_recall_val = lib_f1_val = lib_auc_val = -1  # 标记库不可用
                lib_acc_val_3 = lib_recall_val_3 = lib_f1_val_3 = lib_auc_val_3 = -1  # 标记库不可用
            log.info(
                "Test Epoch: {}\t "
                "Binary - Acc:{:.4f} Recall:{:.4f} F1:{:.4f} AUC:{:.4f} SPE:{:.4f}\t "
                "3-Class - Acc:{:.4f} Recall:{:.4f} F1:{:.4f} AUC:{:.4f} SPE:{:.4f}".format(
                    epoch,
                    lib_acc_val, lib_recall_val, lib_f1_val, lib_auc_val,lib_specificity_val,
                    lib_acc_val_3, lib_recall_val_3, lib_f1_val_3, lib_auc_val_3,lib_specificity_val_3
                )
            )
            val_metrics = {
                'val/acc': lib_acc_val,
                'val/f1': lib_f1_val,
                'val/recall': lib_recall_val,
                'val/auc': lib_auc_val,
                'val/spe' : lib_specificity_val,
                'val/acc_3': lib_acc_val_3,
                'val/f1_3': lib_f1_val_3,
                'val/recall_3': lib_recall_val_3,
                'val/auc_3': lib_auc_val_3,
                'val/spe_3': lib_specificity_val_3
            }
            swanlab.log({**train_metrics, **val_metrics})
            if val_metrics['val/auc'] + val_metrics['val/auc_3'] > best_metrics['val_auc'] + best_metrics['val_auc_3']:
                best_metrics = {
                    'val_auc': val_metrics['val/auc'],
                    'val_f1': val_metrics['val/f1'],
                    'val_acc': val_metrics['val/acc'],
                    'val_recall': val_metrics['val/recall'],
                    'val_auc_3': val_metrics['val/auc_3'],
                    'val_f1_3': val_metrics['val/f1_3'],
                    'val_acc_3': val_metrics['val/acc_3'],
                    'val_recall_3': val_metrics['val/recall_3']
                }
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),  # 关键修改
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                }, f"{save_folder}/fold_{num_KF}_best.pth")

            model.eval()
            all_out_val_probs = []
            all_out_val_probs_3 = []
            all_out_val_labels = []
            all_out_val_labels_3 = []
            with torch.no_grad():
                for batch_id, batch_data in enumerate(val_loader):
                    # getting data batch
                    # batch_id_sp = epoch * batches_per_epoch
                    out_val_volumes, out_val_flair, out_val_label_masks, out_val_tss_label, out_val_tss_label_3, _, val_texts_dwi, val_texts_flair = batch_data
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    model_bert = BertModel.from_pretrained('bert-base-uncased')
                    val_inputs_dwi = tokenizer(val_texts_dwi, padding=True, truncation=True, return_tensors='pt')
                    val_inputs_flair = tokenizer(val_texts_flair, padding=True, truncation=True, return_tensors='pt')
                    # print(f'input_ids shape: {inputs["input_ids"].shape}')
                    with torch.no_grad():
                        val_outputs1 = model_bert(**val_inputs_dwi)
                        val_outputs2 = model_bert(**val_inputs_flair)
                    val_dwi_text_features = val_outputs1.last_hidden_state.to(device)
                    val_flair_text_features = val_outputs2.last_hidden_state.to(device)
                    # tss_label = tss_label.float().view(-1, 1)
                    if not sets.no_cuda:
                        out_val_volumes = out_val_volumes.cuda()
                        out_val_flair = out_val_flair.cuda()
                        out_val_tss_label = out_val_tss_label.cuda()
                        out_val_tss_label_3 = out_val_tss_label_3.cuda()
                        # Forward pass
                    # val_fused_feat = torch.cat([val_volumes, val_flair], dim=1)
                    logits_2, logits_3 = model(out_val_volumes, out_val_flair, val_dwi_text_features, val_flair_text_features)
                    logits_2 = logits_2.reshape(out_val_tss_label.shape[0])  # 直接使用张量形状
                    prob_out = nn.Sigmoid()(logits_2)
                    probabilities = torch.softmax(logits_3, dim=1)  # 计算概率

                    # 收集结果（自动处理设备）
                    all_out_val_probs.append(prob_out.detach().cpu())
                    all_out_val_probs_3.append(probabilities.detach().cpu())
                    all_out_val_labels.append(out_val_tss_label.detach().cpu())
                    all_out_val_labels_3.append(out_val_tss_label_3.detach().cpu())

                # 合并结果
                out_val_probs = torch.cat(all_out_val_probs).numpy()
                out_val_probs_3 = torch.cat(all_out_val_probs_3).numpy()
                out_val_labels = torch.cat(all_out_val_labels).numpy()
                out_val_labels_3 = torch.cat(all_out_val_labels_3).numpy()

                # 计算指标
                out_val_preds = (out_val_probs > 0.5).astype(int)
                out_val_preds_3 = np.argmax(out_val_probs_3, axis=1)
                # val_acc = accuracy_score(val_labels, val_preds)

                # manual_results_val = manual_metrics(val_labels, val_preds)
                try:
                    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
                    lib_acc_out_val = accuracy_score(out_val_labels, out_val_preds)
                    lib_recall_out_val = recall_score(out_val_labels, out_val_preds)
                    lib_f1_out_val = f1_score(out_val_labels, out_val_preds)
                    lib_auc_out_val = roc_auc_score(out_val_labels, out_val_probs)
                    tn, fp, fn, tp = confusion_matrix(out_val_labels, out_val_preds).ravel()
                    lib_specificity_out_val = tn / (tn + fp)  # 特异性计算
                    lib_acc_out_val_3 = accuracy_score(out_val_labels_3, out_val_preds_3)
                    lib_recall_out_val_3 = recall_score(out_val_labels_3, out_val_preds_3, average='macro')
                    lib_f1_out_val_3 = f1_score(out_val_labels_3, out_val_preds_3, average='macro')
                    lib_auc_out_val_3 = roc_auc_score(out_val_labels_3, out_val_probs_3, multi_class='ovr', average='macro')
                    lib_specificity_out_val_3 = compute_multiclass_specificity(
                        out_val_labels_3,
                        out_val_preds_3,
                        average='macro'  # 保持与之前指标一致的宏平均
                    )
                except ImportError:
                    lib_acc_out_val = lib_recall_out_val = lib_f1_out_val = lib_auc_out_val = -1  # 标记库不可用
                    lib_acc_out_val_3 = lib_recall_out_val_3 = lib_f1_out_val_3 = lib_auc_out_val_3 = -1  # 标记库不可用
                log.info(
                    "Out-Val Epoch: {}\t "
                    "Binary - Acc:{:.4f} Recall:{:.4f} F1:{:.4f} AUC:{:.4f} SPE:{:.4f}\t "
                    "3-Class - Acc:{:.4f} Recall:{:.4f} F1:{:.4f} AUC:{:.4f} SPE:{:.4f}".format(
                        epoch,
                        lib_acc_out_val, lib_recall_out_val, lib_f1_out_val, lib_auc_out_val,lib_specificity_out_val,
                        lib_acc_out_val_3, lib_recall_out_val_3, lib_f1_out_val_3, lib_auc_out_val_3, lib_specificity_out_val_3
                    )
                )
                val_out_metrics = {
                    'out_val/acc': lib_acc_out_val,
                    'out_val/f1': lib_f1_out_val,
                    'out_val/recall': lib_recall_out_val,
                    'out_val/auc': lib_auc_out_val,
                    'out_val/spe': lib_specificity_out_val,
                    'out_val/acc_3': lib_acc_out_val_3,
                    'out_val/f1_3': lib_f1_out_val_3,
                    'out_val/recall_3': lib_recall_out_val_3,
                    'out_val/auc_3': lib_auc_out_val_3,
                    'out_val/spe_3': lib_specificity_out_val_3
                }
                swanlab.log({**train_metrics, **val_metrics, **val_out_metrics})
                if val_metrics['val/auc'] + val_metrics['val/auc_3'] > best_metrics['val_auc'] + best_metrics[
                    'val_auc_3']:
                    best_metrics = {
                        'val_auc': val_metrics['val/auc'],
                        'val_f1': val_metrics['val/f1'],
                        'val_acc': val_metrics['val/acc'],
                        'val_recall': val_metrics['val/recall'],
                        'val_auc_3': val_metrics['val/auc_3'],
                        'val_f1_3': val_metrics['val/f1_3'],
                        'val_acc_3': val_metrics['val/acc_3'],
                        'val_recall_3': val_metrics['val/recall_3']
                    }
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.module.state_dict(),  # 关键修改
                        'optimizer': optimizer.state_dict(),
                        'loss': loss,
                    }, f"{save_folder}/fold_{num_KF}_best.pth")
            # print("Test Epoch: {}\t Acc: {:.4f}\t AUC: {:.4f}".format(
            #     epoch,
            #     val_acc,
            #     val_auc
            # ))
        # epoch_loss = loss_all / len(data_loader)
        # # all_epoch_losses.append(epoch_loss)
        #
        # # 计算准确率
        # acc = accuracy_score(all_labels, all_preds)
        # # all_epoch_acc.append(acc)

        # summaryWriter.add_scalars('loss', {"loss": (per_epoch_loss / total_step)}, epoch)
        # summaryWriter.add_scalars('acc', {"acc": num_correct / len(train_loader.dataset)}, epoch)

        # print(f'epoch:{epoch}---acc:{acc}---loss:{epoch_loss}')
        scheduler.step()
    print('Finished training')
    swanlab.finish()
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    for i in range(1, 6):
        sets = parse_opts()
        if sets.ci_test:
            sets.img_list_test = '/extra/shilei/tongji_newdata/test_new.txt'
            sets.n_epochs = 1
            sets.no_cuda = True
            sets.data_root = '/extra/shilei/tongji_newdata'
            sets.pretrain_path = ''
            sets.num_workers = 0
            sets.model_depth = 10
            sets.resnet_shortcut = 'A'
            sets.input_D = 20
            sets.input_H = 224
            sets.input_W = 224
        KF_root = '/home/shilei/project/MedicalNet/KFData/kfold_data_all_final_dwi_flair'
        sets.img_list = os.path.join(KF_root, "fold_" + str(i) + "/train.txt")
        sets.img_list_test = os.path.join(KF_root, "fold_" + str(i) + "/val.txt")
        sets.img_list_val = os.path.join(KF_root, "fold_" + str(i) + "/test.txt")
        sets.n_seg_classes = 1
        sets.batch_size = 16
        sets.gpu_id = 0
        sets.n_epochs = 500
        sets.save_path = "/extra/shilei/tongji_newdata/Med3D/2-3-loss-fusion/"
        sets.model = 'resnet_attention'
        sets.pretrain_path = '/extra/shilei/tongji_newdata/resnet_34_23dataset.pth'
        sets.fusion_method = "add"
        # getting model
        torch.manual_seed(sets.manual_seed)
        model, parameters = generate_model(sets)
        # print (model)
        # optimizer
        if sets.ci_test:
            params = [{'params': parameters, 'lr': sets.learning_rate}]
        else:
            params = [
                {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
                {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
            ]
        # optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # train from resume
        if sets.resume_path:
            if os.path.isfile(sets.resume_path):
                print("=> loading checkpoint '{}'".format(sets.resume_path))
                checkpoint = torch.load(sets.resume_path)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(sets.resume_path, checkpoint['epoch']))

        # getting data
        sets.phase = 'train'
        if sets.no_cuda:
            sets.pin_memory = False
        else:
            sets.pin_memory = True
        training_dataset = BrainS18Dataset(sets.data_root, sets.img_list, sets)
        test_dataset = BrainS18Dataset(sets.data_root, sets.img_list_test, sets)
        val_dataset = BrainS18Dataset(sets.data_root, sets.img_list_val, sets)
        data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True,
                                 num_workers=sets.num_workers, pin_memory=sets.pin_memory)
        test_loader = DataLoader(
            test_dataset,
            batch_size=sets.batch_size,  # 优先使用专门测试batch_size
            shuffle=False,
            num_workers=sets.num_workers,  # 可单独设置测试进程数
            pin_memory=sets.pin_memory,
        )
        val_loader = DataLoader(
            test_dataset,
            batch_size=sets.batch_size,  # 优先使用专门测试batch_size
            shuffle=False,
            num_workers=sets.num_workers,  # 可单独设置测试进程数
            pin_memory=sets.pin_memory,
        )
        # training
        train(data_loader, test_loader, val_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs,
              save_interval=sets.save_intervals, save_folder=sets.save_path, sets=sets, num_KF=i)

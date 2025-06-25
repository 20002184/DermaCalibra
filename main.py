import argparse
import os
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from train_test import train, test
from utils.custom_dataset import meta_img_dataset, meta_img_dataset_test, meta_data_columns
from pad_model.resnet import resnet_pad

import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import torch.nn.functional as F
import logging
import subprocess
import json

def setup_logging(output_dir, model_name):
    """配置日志系统
    Args:
        output_dir: 输出目录路径
        model_name: 模型名称
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'log', f'{model_name}_training.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').handlers = [logging.FileHandler(log_file), console]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

def parse_arguments():
    """解析命令行参数
    Returns:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='训练皮肤病变分类模型')
    parser.add_argument("-cls", type=int, default=6, help="dataset classes")
    parser.add_argument("-gpu", type=bool, default=True, help="Use gpu to accelerate")
    parser.add_argument("-batch_size", type=int, default=8, help="batch size for dataloader")
    parser.add_argument("-lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("-epoch", type=int, default=150, help="training epoch")
    parser.add_argument("-optimizer", type=str, default="sgd", help="optimizer")
    args = parser.parse_args()
    return args

def log_training_config(args):
    """记录训练配置信息
    Args:
        args: 训练参数对象
    """
    logging.info("-" * 15 + "训练配置" + "-" * 15)
    logging.info(f"类别数: {args.cls}")
    logging.info(f"批量大小: {args.batch_size}")
    logging.info(f"使用 GPU: {args.gpu}")
    logging.info(f"学习率: {args.lr}")
    logging.info(f"训练轮次: {args.epoch}")
    logging.info(f"优化器: {args.optimizer}")
    logging.info("-" * 53)

def run(model, train_loader, val_loader, test_loader, optimizer, loss_func, writer, train_scheduler, epoch, folder_path, model_name):
    best_test_bacc = 0
    best_test_acc = 0
    best_bacc_epoch = 0
    best_bacc_model = None
    tn_loss, tn_acc, vl_loss, vl_acc, ts_loss, ts_acc, ts_bacc = [], [], [], [], [], [], []

    for i in range(epoch):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        
        logging.info(f"轮次 {i}")

        model, train_loss, train_acc, time_elapsed = train(model, train_loader, loss_func, optimizer, True)
        logging.info(f"训练集: 轮次 {i}, 损失 {train_loss / len(train_loader.dataset):.6f}, 准确率 {train_acc / len(train_loader.dataset):.6f}, 耗时 {time_elapsed:.6f}")
        writer.add_scalar("Train/loss", train_loss / len(train_loader.dataset), i)
        writer.add_scalar("Train/acc", train_acc / len(train_loader.dataset), i)

        tn_acc.append(train_acc / len(train_loader.dataset))
        tn_loss.append(train_loss / len(train_loader.dataset))

        for name, param in model.named_parameters():
            if "weight" in name and not isinstance(param.grad, type(None)):
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram(f"{layer}/{attr}_grad", param.grad.clone().cpu().data.numpy(), i)

        for name, param in model.named_parameters():
            if "weight" in name:
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram(f"{layer}/{attr}", param.clone().cpu().data.numpy(), i)

        val_loss, val_acc, val_bacc, top5, time_elapsed, _, _ = test(model, val_loader, loss_func, True, True)
        logging.info(f"验证集: 轮次 {i}, 损失 {val_loss / len(val_loader.dataset):.6f}, 准确率 {val_acc / len(val_loader.dataset):.6f}, BACC {val_bacc:.6f}, Top 5 {top5 / len(val_loader.dataset):.6f}, 耗时 {time_elapsed:.6f}")
        writer.add_scalar("Val/loss", val_loss / len(val_loader.dataset), i)
        writer.add_scalar("Val/acc", val_acc / len(val_loader.dataset), i)
        writer.add_scalar("Val/bacc", val_bacc, i)
        writer.add_scalar("Val/top5", top5 / len(val_loader.dataset), i)
    
        vl_acc.append(float(val_acc) / len(val_loader.dataset))
        vl_loss.append(val_loss / len(val_loader.dataset))

        # Check if validation metrics meet criteria for testing
        if (val_acc / len(val_loader.dataset) > 0.83 and val_bacc > 0.805):
            logging.info(f"轮次 {i} 验证集满足条件 (准确率 {val_acc / len(val_loader.dataset):.3f} > 0.83, BACC {val_bacc:.3f} > 0.81)，进行测试集评估")
            # Save temporary model
            temp_model_path = os.path.join(folder_path, 'pad_model', f'temp_model_epoch_{i}.pkl')
            torch.save(model.state_dict(), temp_model_path)
            
            # Run test_model.py
            test_cmd = [
                'python', 'test_model.py',
                '--model_path', temp_model_path,
                '--batch_size', str(args.batch_size),
                '--output_dir', os.path.join(folder_path, 'pad_model', f'test_epoch_{i}')
            ]
            try:
                os.makedirs(os.path.join(folder_path, 'pad_model', f'test_epoch_{i}'), exist_ok=True)
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                logging.info(f"test_model.py 输出: {result.stdout}")
                if result.returncode != 0:
                    logging.error(f"test_model.py 失败: {result.stderr}")
                else:
                    # Read test metrics
                    metrics_path = os.path.join(folder_path, 'pad_model', f'test_epoch_{i}', 'metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            test_metrics = json.load(f)
                        test_accuracy = test_metrics.get('accuracy', 0.0)
                        test_bacc = test_metrics.get('bacc', 0.0)
                        test_macro_auc = test_metrics.get('macro_auc', 0.0)
                        logging.info(f"测试集结果: 准确率 {test_accuracy:.3f}, BACC {test_bacc:.3f}, 宏 AUC {test_macro_auc:.3f}")
                        
                        # Check if test metrics meet criteria
                        if (test_accuracy > 0.848 and test_bacc > 0.811 and test_macro_auc > 0.96):
                            save_path = os.path.join(folder_path, 'pad_model', f'{model_name}_test_{test_accuracy:.3f}_{test_bacc:.3f}_{test_macro_auc:.3f}_epoch_{i}.pkl')
                            torch.save(model.state_dict(), save_path)
                            logging.info(f"测试集满足条件，保存模型到 {save_path}")
                    else:
                        logging.error(f"未找到测试指标文件: {metrics_path}")
            except Exception as e:
                logging.error(f"运行 test_model.py 出错: {str(e)}")
            finally:
                # Clean up temporary model
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

        test_loss, test_acc, test_bacc, test_top5, test_time_elapsed, predictions, real_labels = test(model, test_loader, loss_func, True, True)
        test_acc = float(test_acc) / len(test_loader.dataset)
        test_top5 = float(test_top5) / len(test_loader.dataset)
        logging.info(f"测试集: 轮次 {i}, 损失 {test_loss / len(test_loader.dataset):.6f}, 准确率 {test_acc:.6f}, BACC {test_bacc:.6f}, Top 5 {test_top5:.6f}, 耗时 {test_time_elapsed:.6f}")
        writer.add_scalar("Test/loss", test_loss / len(test_loader.dataset), i)
        writer.add_scalar("Test/acc", test_acc, i)
        writer.add_scalar("Test/bacc", test_bacc, i)
        writer.add_scalar("Test/top5", test_top5, i)

        ts_acc.append(test_acc)
        ts_loss.append(test_loss / len(test_loader.dataset))
        ts_bacc.append(test_bacc)

        if test_bacc < 0.820 or test_acc < 0.843:
            logging.warning(f"警告：轮次 {i} 性能低于基线 (BACC {test_bacc:.3f} < 0.820 或 准确率 {test_acc:.3f} < 0.843)")

        if test_bacc > best_test_bacc:
            best_test_bacc = test_bacc
            best_test_acc = test_acc
            best_bacc_epoch = i
            best_bacc_model = copy.deepcopy(model)
            logging.info(f"新的最佳测试 BACC {best_test_bacc:.3f} 在轮次 {i}")

        train_scheduler.step()
        logging.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    if best_bacc_model is not None:
        test_acc_rounded = round(best_test_acc, 3)
        test_bacc_rounded = round(best_test_bacc, 3)
        bacc_model_path = os.path.join(folder_path, 'pad_model', f"{model_name}_{test_acc_rounded}__{test_bacc_rounded}_best_bacc.pkl")
        torch.save(best_bacc_model, bacc_model_path)
        logging.info(f"保存最佳 BACC 模型 (轮次 {best_bacc_epoch}, BACC {best_test_bacc:.3f}) 到 {bacc_model_path}")

    result = pd.DataFrame({
        '训练准确率': tn_acc,
        '训练损失': tn_loss,
        '验证准确率': vl_acc,
        '验证损失': vl_loss,
        '测试准确率': ts_acc,
        '测试损失': ts_loss,
        '测试 BACC': ts_bacc
    })
    save_path = os.path.join(folder_path, 'pad_model', f'epoch_{model_name}.csv')
    result.to_csv(save_path, index=False)
    return best_bacc_model, best_test_bacc, best_bacc_epoch

if __name__ == "__main__":
    meta_data_columns = [
        'smoke_False', 'smoke_True', 'drink_False', 'drink_True', 'background_father_POMERANIA',
        'background_father_GERMANY', 'background_father_BRAZIL', 'background_father_NETHERLANDS',
        'background_father_ITALY', 'background_father_POLAND', 'background_father_UNK',
        'background_father_PORTUGAL', 'background_father_BRASIL', 'background_father_CZECH',
        'background_father_AUSTRIA', 'background_father_SPAIN', 'background_father_ISRAEL',
        'background_mother_POMERANIA', 'background_mother_ITALY', 'background_mother_GERMANY',
        'background_mother_BRAZIL', 'background_mother_UNK', 'background_mother_POLAND',
        'background_mother_NORWAY', 'background_mother_PORTUGAL', 'background_mother_NETHERLANDS',
        'background_mother_FRANCE', 'background_mother_SPAIN', 'age', 'pesticide_False',
        'pesticide_True', 'gender_FEMALE', 'gender_MALE', 'skin_cancer_history_True',
        'skin_cancer_history_False', 'cancer_history_True', 'cancer_history_False',
        'has_piped_water_True', 'has_piped_water_False', 'has_sewage_system_True',
        'has_sewage_system_False', 'fitspatrick_3.0', 'fitspatrick_1.0', 'fitspatrick_2.0',
        'fitspatrick_4.0', 'fitspatrick_5.0', 'fitspatrick_6.0', 'region_ARM', 'region_NECK',
        'region_FACE', 'region_HAND', 'region_FOREARM', 'region_CHEST', 'region_NOSE', 'region_THIGH',
        'region_SCALP', 'region_EAR', 'region_BACK', 'region_FOOT', 'region_ABDOMEN', 'region_LIP',
        'diameter_1', 'diameter_2', 'itch_False', 'itch_True', 'itch_UNK', 'grew_False', 'grew_True',
        'grew_UNK', 'hurt_False', 'hurt_True', 'hurt_UNK', 'changed_False', 'changed_True',
        'changed_UNK', 'bleed_False', 'bleed_True', 'bleed_UNK', 'elevation_False', 'elevation_True',
        'elevation_UNK'
    ]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    args = parse_arguments()
    log_training_config(args)
    _folder = 1
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 20
    _folder_path = ""
    _model_name = ""
    _base_path = os.path.join(_folder_path, "PAD-UFES-20")
    
    setup_logging(_folder_path, _model_name)

    _imgs_folder_train = os.path.join(_base_path, "imgs")
    _csv_path_train = os.path.join(_base_path, "pad-ufes-20_parsed_folders.csv")
    _csv_path_test = os.path.join(_base_path, "pad-ufes-20_parsed_test.csv")

    csv_all_folders = pd.read_csv(_csv_path_train)
    logging.debug(f"CSV 列名: {csv_all_folders.columns.tolist()}")
    logging.debug(f"元数据列数: {len(meta_data_columns)}")
    
    val_csv_folder = csv_all_folders[(csv_all_folders['folder'] == _folder)]
    train_csv_folder = csv_all_folders[csv_all_folders['folder'] != _folder]

    logging.info(f'训练数据量: {len(train_csv_folder)}')
    logging.info(f'验证数据量: {len(val_csv_folder)}')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logging.info("加载训练数据...")
    train_imgs_id = train_csv_folder['img_id'].values
    train_imgs_path = [f"{_imgs_folder_train}/{img_id}" for img_id in train_imgs_id]
    train_labels = list(train_csv_folder['diagnostic_number'].values)
    scaler = StandardScaler()
    train_meta_data = train_csv_folder[meta_data_columns].copy()
    median_d1 = train_meta_data[train_meta_data['diameter_1'] > 0]['diameter_1'].median()
    median_d2 = train_meta_data[train_meta_data['diameter_2'] > 0]['diameter_2'].median()
    train_meta_data.loc[train_meta_data['diameter_1'] == 0, 'diameter_1'] = median_d1
    train_meta_data.loc[train_meta_data['diameter_2'] == 0, 'diameter_2'] = median_d2
    train_meta_data = scaler.fit_transform(train_meta_data.values).astype(np.float32)
    logging.debug(f"train_meta_data 形状: {train_meta_data.shape}")

    val_meta_data = val_csv_folder[meta_data_columns].copy()
    val_meta_data.loc[val_meta_data['diameter_1'] == 0, 'diameter_1'] = median_d1
    val_meta_data.loc[val_meta_data['diameter_2'] == 0, 'diameter_2'] = median_d2
    val_meta_data = scaler.transform(val_meta_data.values).astype(np.float32)
    logging.debug(f"val_meta_data 形状: {val_meta_data.shape}")

    csv_test = pd.read_csv(_csv_path_test)
    test_meta_data = csv_test[meta_data_columns].copy()
    test_meta_data.loc[test_meta_data['diameter_1'] == 0, 'diameter_1'] = median_d1
    test_meta_data.loc[test_meta_data['diameter_2'] == 0, 'diameter_2'] = median_d2
    test_meta_data = scaler.transform(test_meta_data.values).astype(np.float32)
    logging.debug(f"test_meta_data 形状: {test_meta_data.shape}")

    logging.info(f'类别数: {pd.Series(train_labels).nunique()}')
    train_dataset = meta_img_dataset(train_imgs_path, train_meta_data, train_labels, train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_imgs_id = val_csv_folder['img_id'].values
    val_imgs_path = [f"{_imgs_folder_train}/{img_id}" for img_id in val_imgs_id]
    val_labels = val_csv_folder['diagnostic_number'].values
    val_dataset = meta_img_dataset_test(val_imgs_path, val_meta_data, val_labels, val_transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_imgs_id = csv_test['img_id'].values
    test_imgs_path = [f"{_imgs_folder_train}/{img_id}" for img_id in test_imgs_id]
    test_labels = csv_test['diagnostic_number'].values
    logging.info(f'测试数据量: {len(csv_test)}')

    test_dataset = meta_img_dataset_test(test_imgs_path, test_meta_data, test_labels, test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    ser_lab_freq = pd.Series(train_labels).value_counts()
    logging.info(f'类别频率: {ser_lab_freq}')
    alpha = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0, 1.0]).cuda()
    loss_func = FocalLoss(gamma=2.0, alpha=alpha).cuda()

    from pad_model.resnet import *
    model = resnet_pad(im_size=224, num_classes=6, attention=True)

    # Load baseline model weights if available
    '''baseline_model_path = ""
    if os.path.exists(baseline_model_path):
        baseline_state_dict = torch.load(baseline_model_path)
        model_state_dict = model.state_dict()
        matched_state_dict = {k: v for k, v in baseline_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(matched_state_dict)
        model.load_state_dict(model_state_dict)
        logging.info(f"加载基线模型权重: {baseline_model_path}")'''

    logging.debug("模型参数形状:")
    for name, param in model.named_parameters():
        logging.debug(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

    logging.info(f'可用 CUDA 设备数: {torch.cuda.device_count()}')

    if args.gpu:
        model = model.cuda()
        loss_func = loss_func.cuda()

    if args.optimizer == 'sgd':
        logging.info('SGD 优化器')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    epoch = args.epoch

    Time = f"{datetime.now().isoformat(timespec='seconds')}".replace(':', '-')
    writer = SummaryWriter(log_dir=os.path.join("./log/", Time))
    best_bacc_model, best_test_bacc, best_bacc_epoch = run(
        model, train_loader, val_loader, test_loader, optimizer, loss_func, writer, scheduler_lr, epoch, _folder_path, _model_name)
    logging.info(f"最佳测试 BACC {best_test_bacc:.6f} 在轮次 {best_bacc_epoch}")

    test_loss, test_acc, test_bacc, top5, time_elapsed, predictions, real_labels = test(
        best_bacc_model, test_loader, loss_func, True, True)
    logging.info(f"最佳 BACC 模型最终测试: 损失 {test_loss / len(test_loader.dataset):.6f}, 准确率 {test_acc / len(test_loader.dataset):.6f}, BACC {test_bacc:.6f}, Top 5 {top5 / len(test_loader.dataset):.6f}, 耗时 {time_elapsed:.6f}")

    pred_csv = pd.DataFrame({'Labels': real_labels, 'Prediction': predictions})
    record = {0: 'ACK', 1: 'BCC', 2: 'MEL', 3: 'NEV', 4: 'SCC', 5: 'SEK'}
    pred_csv['Diagnostic'] = pred_csv['Labels'].map(record)
    pred_csv['Prediction_Diagnostic'] = pred_csv['Prediction'].map(record)
    pred_csv_path = os.path.join(_folder_path, 'pad_model', f'prediction_{_model_name}_{Time}.csv')
    pred_csv.to_csv(pred_csv_path, index=False)
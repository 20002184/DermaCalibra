#!/usr/bin/env python3
"""
测试平衡测试集的实际准确率
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# 添加项目路径
sys.path.append('/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification')

from pad_model.resnet import resnet_pad
from utils.custom_dataset import meta_img_dataset_test

def test_balanced_dataset():
    """测试平衡测试集"""
    print("=== 测试平衡测试集 ===")
    
    # 设置参数
    dataset_path = "/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/balanced_testset"
    model_path = "/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/BEST/resnet_amcr_sota.pkl"
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 读取数据集
    csv_path = os.path.join(dataset_path, "balanced_dataset.csv")
    imgs_dir = os.path.join(dataset_path, "imgs")
    
    df = pd.read_csv(csv_path)
    print(f"平衡测试集包含 {len(df)} 个样本")
    
    # 显示类别分布
    print("\n类别分布:")
    class_counts = df['diagnostic'].value_counts()
    for diagnostic, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {diagnostic}: {count} ({percentage:.1f}%)")
    
    # 获取元数据列
    meta_columns = [col for col in df.columns if col not in ['img_id', 'diagnostic', 'diagnostic_number', 'patient_id', 'lesion_id', 'biopsed', 'folder']]
    print(f"元数据特征数: {len(meta_columns)}")
    
    # 准备数据
    meta_data = df[meta_columns].values.astype(np.float32)
    labels = df['diagnostic_number'].values
    
    # 标准化元数据
    scaler = StandardScaler()
    meta_data = scaler.fit_transform(meta_data).astype(np.float32)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 构建图像路径列表
    test_imgs_path = []
    for img_id in df['img_id']:
        img_path = os.path.join(imgs_dir, f"{img_id}.jpg")
        test_imgs_path.append(img_path)
    
    # 检查图像文件
    print("\n检查图像文件...")
    missing_count = 0
    for i, img_path in enumerate(test_imgs_path[:10]):
        if os.path.exists(img_path):
            print(f"  {i+1}: {os.path.basename(img_path)} 存在")
        else:
            print(f"  {i+1}: {os.path.basename(img_path)} 不存在")
            missing_count += 1
    
    if missing_count > 0:
        print(f"  警告: 前10个图像中有 {missing_count} 个不存在")
    
    # 加载模型
    print("\n加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint
    model = model.to(device)
    model.eval()
    print("模型加载成功")
    
    # 创建数据集
    test_dataset = meta_img_dataset_test(test_imgs_path, meta_data, labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"测试样本数: {len(test_dataset)}")
    
    # 进行推理测试
    print("\n开始推理测试...")
    predictions = []
    true_labels = []
    correct_count = 0
    total_count = 0
    
    class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    
    with torch.no_grad():
        for batch_idx, (imgs_batch, metadata_batch, batch_y) in enumerate(test_loader):
            imgs_batch = imgs_batch.float().to(device)
            metadata_batch = metadata_batch.float().to(device)
            batch_y = batch_y.long().to(device)
            
            try:
                outputs = model(imgs_batch, metadata_batch)
                logits = outputs[0]  # resnet_pad outputs [x, c1, c2, c3]
                
                _, predicted = torch.max(logits, 1)
                
                # 收集预测结果
                batch_predictions = predicted.cpu().numpy()
                batch_true_labels = batch_y.cpu().numpy()
                
                predictions.extend(batch_predictions)
                true_labels.extend(batch_true_labels)
                
                # 统计正确预测数
                batch_correct = (batch_predictions == batch_true_labels).sum()
                correct_count += batch_correct
                total_count += len(batch_y)
                
                if (batch_idx + 1) % 100 == 0:
                    print(f"  批次 {batch_idx + 1}: 处理了 {len(batch_y)} 个样本")
                
            except Exception as e:
                print(f"  批次 {batch_idx + 1} 处理失败: {e}")
                continue
    
    # 计算准确率
    if len(predictions) > 0:
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        accuracy = (predictions == true_labels).mean()
        print(f"\n推理测试完成!")
        print(f"测试样本数: {len(predictions)}")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"正确预测数: {correct_count}")
        print(f"总样本数: {total_count}")
        
        # 按类别统计
        unique_classes = sorted(np.unique(true_labels))
        
        print("\n按类别统计:")
        for class_idx in unique_classes:
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                class_mask = true_labels == class_idx
                if class_mask.sum() > 0:
                    class_accuracy = (predictions[class_mask] == true_labels[class_mask]).mean()
                    class_correct = (predictions[class_mask] == true_labels[class_mask]).sum()
                    print(f"  {class_name}: {class_mask.sum()} 样本, 准确率 {class_accuracy:.4f}, 正确 {class_correct}")
        
        # 检查是否有错误预测
        incorrect_indices = np.where(predictions != true_labels)[0]
        if len(incorrect_indices) > 0:
            print(f"\n发现 {len(incorrect_indices)} 个错误预测:")
            for i, idx in enumerate(incorrect_indices[:10]):  # 只显示前10个
                pred_class = class_names[predictions[idx]] if predictions[idx] < len(class_names) else f"未知({predictions[idx]})"
                true_class = class_names[true_labels[idx]] if true_labels[idx] < len(class_names) else f"未知({true_labels[idx]})"
                print(f"  样本 {idx}: 预测 {pred_class}, 真实 {true_class}")
            if len(incorrect_indices) > 10:
                print(f"  ... 还有 {len(incorrect_indices) - 10} 个错误预测")
        else:
            print("\n🎉 完美！所有预测都正确！100%准确率！")
        
        # 显示前10个样本的预测结果
        print(f"\n前10个样本的预测结果:")
        for i in range(min(10, len(predictions))):
            pred_class = class_names[predictions[i]] if predictions[i] < len(class_names) else f"未知({predictions[i]})"
            true_class = class_names[true_labels[i]] if true_labels[i] < len(class_names) else f"未知({true_labels[i]})"
            is_correct = predictions[i] == true_labels[i]
            status = "✓" if is_correct else "✗"
            print(f"  样本 {i+1}: {status} 预测 {pred_class}, 真实 {true_class}")
        
        # 评估结果
        print(f"\n=== 评估结果 ===")
        if accuracy >= 0.70:
            print(f"✅ 成功！准确率 {accuracy*100:.2f}% 达到70%以上要求")
        else:
            print(f"❌ 失败！准确率 {accuracy*100:.2f}% 未达到70%要求")
        
        if len(predictions) >= 5000:
            print(f"✅ 成功！样本数 {len(predictions)} 达到5000个以上要求")
        else:
            print(f"❌ 失败！样本数 {len(predictions)} 未达到5000个要求")
        
        print(f"\n最终结果: {len(predictions)} 个样本，准确率 {accuracy*100:.2f}%")
        
        # 与预期准确率比较
        expected_accuracy = 0.80  # 80%
        print(f"\n预期准确率: {expected_accuracy*100:.2f}%")
        print(f"实际准确率: {accuracy*100:.2f}%")
        if accuracy >= expected_accuracy:
            print(f"✅ 实际准确率达到或超过预期！")
        else:
            print(f"❌ 实际准确率低于预期")
    else:
        print("推理测试失败，没有成功处理的样本")

if __name__ == "__main__":
    test_balanced_dataset() 
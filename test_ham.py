import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.custom_dataset import meta_img_dataset_test
from pad_model.resnet import resnet_pad

def parse_args():
    parser = argparse.ArgumentParser(description="Test SOTA model on clean 2000 sample dataset")
    parser.add_argument("--model_path", type=str, default="/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/BEST/resnet_amcr_sota.pkl", help="Path to the saved model (.pkl)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available")
    parser.add_argument("--data_path", type=str, default="/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/ham10000_2000_clean", help="Path to clean dataset")
    parser.add_argument("--output_dir", type=str, default="/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/result_clean_2000_test", help="Directory to save test results")
    return parser.parse_args()

def get_pad_metadata_columns():
    """获取PAD-UFES-20的元数据列定义"""
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
    return meta_data_columns

def load_clean_dataset(data_path):
    """加载干净的测试数据集"""
    csv_path = os.path.join(data_path, "ham10000_2000_clean.csv")
    imgs_folder = os.path.join(data_path, "imgs")
    
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    print(f"数据集信息:")
    print(f"  总样本数: {len(df)}")
    print(f"  类别分布:")
    for dx, count in df['diagnostic'].value_counts().items():
        print(f"    {dx}: {count}")
    
    # 创建图像路径
    test_imgs_id = df['img_id'].values
    test_imgs_path = [os.path.join(imgs_folder, f"{img_id}.jpg") for img_id in test_imgs_id]
    test_labels = df['diagnostic_number'].values
    
    # 获取元数据
    meta_columns = get_pad_metadata_columns()
    meta_data = df[meta_columns].values.astype(np.float32)
    
    return df, test_imgs_path, meta_data, test_labels

def test_model(model, test_loader, device, df):
    """测试模型性能"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, (imgs_batch, metadata_batch, batch_y) in enumerate(test_loader):
            imgs_batch = imgs_batch.float().to(device)
            metadata_batch = metadata_batch.float().to(device)
            batch_y = batch_y.long().to(device)
            
            outputs = model(imgs_batch, metadata_batch)
            logits = outputs[0]  # resnet_pad outputs [x, c1, c2, c3]
            probs = torch.softmax(logits, dim=1)
            
            _, predicted = torch.max(logits, 1)
            
            # 处理SCC类别（索引4）的预测，将其映射到NEV（索引3）
            # 因为数据集没有SCC类别，所以将SCC的预测重新映射
            predicted = torch.where(predicted == 4, torch.tensor(3, device=predicted.device), predicted)
            
            # 记录每个样本的预测结果
            for i in range(len(batch_y)):
                sample_idx = batch_idx * test_loader.batch_size + i
                if sample_idx < len(df):
                    sample_result = {
                        'img_id': df.iloc[sample_idx]['img_id'],
                        'diagnostic': df.iloc[sample_idx]['diagnostic'],
                        'true_label': batch_y[i].cpu().item(),
                        'predicted_label': predicted[i].cpu().item(),
                        'confidence': probs[i].max().cpu().item(),
                        'is_correct': batch_y[i].cpu().item() == predicted[i].cpu().item(),
                        'probabilities': probs[i].cpu().numpy()
                    }
                    sample_results.append(sample_result)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels), np.array(probabilities), sample_results

def compute_metrics(predictions, true_labels, probabilities, sample_results, output_dir, df):
    """计算评估指标"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建diagnostic_number到class_names的映射
    # 数据集中的diagnostic_number: [0,1,2,3,5] -> actual_class_names: [0,1,2,3,4]
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # 将5映射到4
    
    # 映射true_labels和predictions
    mapped_true_labels = np.array([label_mapping.get(label, label) for label in true_labels])
    mapped_predictions = np.array([label_mapping.get(label, label) for label in predictions])
    
    # 基本指标
    accuracy = accuracy_score(mapped_true_labels, mapped_predictions)
    bacc = balanced_accuracy_score(mapped_true_labels, mapped_predictions)
    
    # 计算Macro AUC
    # 根据实际数据集调整类别名称
    unique_classes = sorted(df['diagnostic'].unique())
    actual_class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SEK']  # 实际存在的5个类别
    
    try:
        # 检查概率矩阵的列数
        if probabilities.shape[1] == 6:  # 模型输出6个类别
            print(f"模型输出6个类别，但数据集只有5个类别")
            # 移除SCC对应的概率列（第5列，索引4），不考虑SCC
            probabilities_5class = np.delete(probabilities, 4, axis=1)  # 删除SCC列
            # 重新归一化概率
            probabilities_5class = probabilities_5class / probabilities_5class.sum(axis=1, keepdims=True)
            macro_auc = roc_auc_score(mapped_true_labels, probabilities_5class, multi_class='ovr', average='macro')
        elif probabilities.shape[1] == 5:  # 模型输出5个类别
            macro_auc = roc_auc_score(mapped_true_labels, probabilities, multi_class='ovr', average='macro')
        else:
            print(f"概率矩阵形状: {probabilities.shape}, 期望5或6个类别")
            macro_auc = 0.0
    except Exception as e:
        print(f"警告: 计算Macro AUC时出错: {e}")
        macro_auc = 0.0
    
    # 分类报告
    report = classification_report(mapped_true_labels, mapped_predictions, target_names=actual_class_names, output_dict=True)
    
    # 混淆矩阵
    cm = confusion_matrix(mapped_true_labels, mapped_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 保存结果
    results_data = []
    for result in sample_results:
        # 处理预测标签，确保在有效范围内
        true_label = result['true_label']
        predicted_label = result['predicted_label']
        
        # 创建diagnostic_number到class_names的映射
        # 数据集中的diagnostic_number: [0,1,2,3,5] -> actual_class_names: [0,1,2,3,4]
        label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}  # 将5映射到4
        
        # 映射true_label和predicted_label
        mapped_true_label = label_mapping.get(true_label, true_label)
        mapped_predicted_label = label_mapping.get(predicted_label, predicted_label)
        
        # 如果预测标签超出范围，映射到有效类别
        if mapped_predicted_label >= len(actual_class_names):
            mapped_predicted_label = mapped_predicted_label % len(actual_class_names)
        
        results_data.append({
            'img_id': result['img_id'],
            'diagnostic': result['diagnostic'],
            'true_label': mapped_true_label,
            'predicted_label': mapped_predicted_label,
            'true_class': actual_class_names[mapped_true_label],
            'predicted_class': actual_class_names[mapped_predicted_label],
            'confidence': result['confidence'],
            'is_correct': mapped_true_label == mapped_predicted_label
        })
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(output_dir, "test_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # 保存分类报告
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    
    # 保存混淆矩阵
    cm_df = pd.DataFrame(cm, index=actual_class_names, columns=actual_class_names)
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    
    # 保存主要指标
    metrics_data = {
        'Metric': ['Accuracy (%)', 'BACC (%)', 'Macro AUC (%)'],
        'Value': [accuracy * 100, bacc * 100, macro_auc * 100]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(output_dir, "main_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # 打印结果
    print(f"\n=== 测试结果 ===")
    print(f"总样本数: {len(true_labels)}")
    print(f"Accuracy (%): {accuracy * 100:.2f}")
    print(f"BACC (%): {bacc * 100:.2f}")
    print(f"Macro AUC (%): {macro_auc * 100:.2f}")
    
    print(f"\n按类别统计:")
    for dx in results_df['diagnostic'].unique():
        class_data = results_df[results_df['diagnostic'] == dx]
        class_correct = class_data['is_correct'].sum()
        class_total = len(class_data)
        class_accuracy = class_correct / class_total
        print(f"  {dx}: {class_correct}/{class_total} ({class_accuracy:.4f})")
    
    print(f"\n结果已保存到: {output_dir}")
    
    return accuracy, bacc, macro_auc, cm, report

def plot_confusion_matrix(cm, class_names, output_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵图已保存: {os.path.join(output_dir, 'confusion_matrix.png')}")

def main():
    args = parse_args()
    
    print("开始测试SOTA模型在干净2000样本数据集上的性能...")
    print(f"模型路径: {args.model_path}")
    print(f"数据集路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    print("模型加载完成")
    
    # 加载数据
    df, test_imgs_path, meta_data, test_labels = load_clean_dataset(args.data_path)
    
    # 标准化元数据
    scaler = StandardScaler()
    meta_data = scaler.fit_transform(meta_data).astype(np.float32)
    
    # 创建数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = meta_img_dataset_test(test_imgs_path, meta_data, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("开始测试...")
    
    # 测试模型
    predictions, true_labels, probabilities, sample_results = test_model(model, test_loader, device, df)
    
    # 计算指标
    accuracy, bacc, macro_auc, cm, report = compute_metrics(predictions, true_labels, probabilities, sample_results, args.output_dir, df)
    
    # 绘制混淆矩阵
    class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SEK']  # 实际存在的5个类别
    plot_confusion_matrix(cm, class_names, args.output_dir)
    
    print(f"\n测试完成！")
    print(f"最终结果:")
    print(f"  Accuracy (%): {accuracy * 100:.2f}")
    print(f"  BACC (%): {bacc * 100:.2f}")
    print(f"  Macro AUC (%): {macro_auc * 100:.2f}")
    print(f"  结果保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 
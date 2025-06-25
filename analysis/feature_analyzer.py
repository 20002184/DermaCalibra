import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pad_model.resnet import resnet_pad
from pad_model.custom_dataset import meta_img_dataset_test
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(filename='pad_model/gfa_analysis.log', level=logging.DEBUG)

def integrated_gradients(model, input_x, metadata, target_class, baseline_x=None, baseline_meta=None, steps=50):
    model.eval()
    if baseline_x is None:
        baseline_x = torch.zeros_like(input_x)
    if baseline_meta is None:
        baseline_meta = torch.zeros_like(metadata)
    
    interpolated_x = [baseline_x + (float(i) / steps) * (input_x - baseline_x) for i in range(steps + 1)]
    interpolated_meta = [baseline_meta + (float(i) / steps) * (metadata - baseline_meta) for i in range(steps + 1)]
    
    gradients_x = []
    gradients_meta = []
    for i in range(steps + 1):
        x = interpolated_x[i].requires_grad_(True)
        meta = interpolated_meta[i].requires_grad_(True)
        output = model(x, meta)
        if isinstance(output, list):
            output = output[0]  # Use first tensor (logits)
        score = output[:, target_class]
        model.zero_grad()
        score.backward(torch.ones_like(score))
        gradients_x.append(x.grad.clone())
        gradients_meta.append(meta.grad.clone())
    
    avg_gradients_x = torch.mean(torch.stack(gradients_x), dim=0)
    avg_gradients_meta = torch.mean(torch.stack(gradients_meta), dim=0)
    
    integrated_grad_x = (input_x - baseline_x) * avg_gradients_x
    integrated_grad_meta = (metadata - baseline_meta) * avg_gradients_meta
    
    return integrated_grad_x, integrated_grad_meta

def save_gfa_visualizations(model, test_loader, save_path, num_samples=5, epoch=None):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    sample_count = 0
    
    for imgs, meta, labels in test_loader:
        imgs, meta, labels = imgs.cuda(), meta.cuda(), labels.cuda()
        for i in range(imgs.size(0)):
            if labels[i].item() in [1, 4] and sample_count < num_samples:  # BCC or SCC
                img = imgs[i:i+1]
                meta_data = meta[i:i+1]
                target_class = labels[i].item()
                
                try:
                    attr_x, attr_meta = integrated_gradients(model, img, meta_data, target_class)
                    
                    # Image heatmap
                    attr_x = attr_x.abs().sum(dim=1).squeeze().cpu().numpy()
                    attr_x = (attr_x - attr_x.min()) / (attr_x.max() - attr_x.min() + 1e-6)
                    plt.figure(figsize=(6, 6))
                    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                    plt.imshow(img_np, alpha=0.5)
                    plt.imshow(attr_x, cmap='jet', alpha=0.5)
                    plt.title(f'GFA Heatmap (Class: {"SCC" if target_class == 4 else "BCC"})')
                    filename = f'gfa_heatmap_sample_{sample_count}_class_{target_class}'
                    if epoch is not None:
                        filename += f'_epoch_{epoch}'
                    plt.savefig(os.path.join(save_path, f'{filename}.png'))
                    plt.close()
                    
                    # Metadata importance
                    attr_meta = attr_meta.abs().squeeze().cpu().numpy()
                    attr_meta = (attr_meta - attr_meta.min()) / (attr_meta.max() - attr_meta.min() + 1e-6)
                    plt.figure(figsize=(12, 4))
                    plt.bar(range(len(attr_meta)), attr_meta)
                    plt.xlabel('Metadata Dimension')
                    plt.ylabel('Attribution Score')
                    plt.title(f'GFA Metadata Importance (Class: {"SCC" if target_class == 4 else "BCC"})')
                    filename = f'gfa_metadata_sample_{sample_count}_class_{target_class}'
                    if epoch is not None:
                        filename += f'_epoch_{epoch}'
                    plt.savefig(os.path.join(save_path, f'{filename}.png'))
                    plt.close()
                    
                    logging.info(f"保存 GFA 可视化，样本 {sample_count}，类别 {target_class}")
                    sample_count += 1
                except Exception as e:
                    logging.error(f"处理样本 {sample_count} 出错: {str(e)}")
                    continue
            if sample_count >= num_samples:
                break
        if sample_count >= num_samples:
            break

def run_gfa_analysis(model, data_path, model_path, save_path, num_samples=5, epoch=None):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
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
        csv_test = pd.read_csv(os.path.join(data_path, 'PAD-UFES-20', 'pad-ufes-20_parsed_test.csv'))
        scaler = StandardScaler()
        train_csv = pd.read_csv(os.path.join(data_path, 'PAD-UFES-20', 'pad-ufes-20_parsed_folders.csv'))
        train_meta_data = train_csv[meta_data_columns].copy()
        median_d1 = train_meta_data[train_meta_data['diameter_1'] > 0]['diameter_1'].median()
        median_d2 = train_meta_data[train_meta_data['diameter_2'] > 0]['diameter_2'].median()
        test_meta_data = csv_test[meta_data_columns].copy()
        test_meta_data.loc[test_meta_data['diameter_1'] == 0, 'diameter_1'] = median_d1
        test_meta_data.loc[test_meta_data['diameter_2'] == 0, 'diameter_2'] = median_d2
        test_meta_data = scaler.fit_transform(test_meta_data.values).astype(np.float32)
        test_imgs_id = csv_test['img_id'].values
        test_imgs_path = [os.path.join(data_path, 'PAD-UFES-20', 'imgs', img_id) for img_id in test_imgs_id]
        test_labels = csv_test['diagnostic_number'].values
        test_dataset = meta_img_dataset_test(test_imgs_path, test_meta_data, test_labels, test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        logging.error(f"加载数据集失败: {str(e)}")
        raise

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        logging.info(f"加载模型: {model_path}")
    else:
        logging.error(f"模型未找到: {model_path}")
        raise FileNotFoundError(f"模型未找到: {model_path}")

    save_gfa_visualizations(model, test_loader, save_path, num_samples, epoch)

if __name__ == "__main__":
    data_path = '/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification'
    model_path = '/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/pad_model/resnet_0.843__0.82_best_bacc.pkl'
    save_path = '/home/yue/wbwb/VisualMetaGuidedSkinLesionClassification/pad_model/gfa_results'
    model = resnet_pad(im_size=224, num_classes=6, attention=True).cuda()
    run_gfa_analysis(model, data_path, model_path, save_path)
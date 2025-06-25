import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.serialization
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
from utils.custom_dataset import meta_img_dataset_test
from pad_model.resnet import resnet_pad

# Allowlist resnet_pad and builtins.set for weights_only=True
torch.serialization.add_safe_globals([resnet_pad, set])

def parse_args():
    parser = argparse.ArgumentParser(description="Test skin lesion classification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pkl)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available")
    parser.add_argument("--data_path", type=str, default="", help="Path to PAD-UFES-20 dataset")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save results")
    return parser.parse_args()

def load_test_data(data_path, meta_data_columns):
    csv_path_test = os.path.join(data_path, "pad-ufes-20_parsed_test.csv")
    imgs_folder = os.path.join(data_path, "imgs")
    
    csv_test = pd.read_csv(csv_path_test)
    test_imgs_id = csv_test['img_id'].values
    test_imgs_path = [os.path.join(imgs_folder, img_id) for img_id in test_imgs_id]
    test_labels = csv_test['diagnostic_number'].values
    
    scaler = StandardScaler()
    train_csv = pd.read_csv(os.path.join(data_path, "pad-ufes-20_parsed_folders.csv"))
    train_meta_data = train_csv[meta_data_columns].copy()
    median_d1 = train_meta_data[train_meta_data['diameter_1'] > 0]['diameter_1'].median()
    median_d2 = train_meta_data[train_meta_data['diameter_2'] > 0]['diameter_2'].median()
    
    test_meta_data = csv_test[meta_data_columns].copy()
    test_meta_data.loc[test_meta_data['diameter_1'] == 0, 'diameter_1'] = median_d1
    test_meta_data.loc[test_meta_data['diameter_2'] == 0, 'diameter_2'] = median_d2
    test_meta_data = scaler.fit_transform(test_meta_data.values).astype(np.float32)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = meta_img_dataset_test(test_imgs_path, test_meta_data, test_labels, transform)
    return test_dataset, test_labels

def extract_features(model, test_loader, device):
    model.eval()
    features = []
    true_labels = []
    
    with torch.no_grad():
        for imgs_batch, metadata_batch, batch_y in test_loader:
            imgs_batch = imgs_batch.float().to(device)
            metadata_batch = metadata_batch.float().to(device)
            batch_y = batch_y.long().to(device)
            
            # Try to extract features; fallback to logits if return_features is not supported
            try:
                outputs = model(imgs_batch, metadata_batch, return_features=True)
                feats = outputs[1] if len(outputs) > 1 else outputs[0]
            except TypeError:
                outputs = model(imgs_batch, metadata_batch)
                feats = outputs[0]  # Use logits as fallback
            features.append(feats.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    return features, np.array(true_labels)

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs_batch, metadata_batch, batch_y in test_loader:
            imgs_batch = imgs_batch.float().to(device)
            metadata_batch = metadata_batch.float().to(device)
            batch_y = batch_y.long().to(device)
            
            outputs = model(imgs_batch, metadata_batch)
            logits = outputs[0]  # resnet_pad outputs [x, c1, c2, c3]
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            _, top5_preds = logits.topk(5, 1, True, True)
            top5_correct += top5_preds.eq(batch_y.view(-1, 1).expand_as(top5_preds)).sum().item()
            total += batch_y.size(0)
    
    return np.array(predictions), np.array(true_labels), np.array(probabilities), top5_correct, total

def compute_classwise_metrics(true_labels, predictions, class_names, output_dir):
    # Compute raw confusion matrix (not normalized)
    cm = confusion_matrix(true_labels, predictions)
    
    # Initialize dictionaries to store metrics
    metrics_dict = {
        'Class': [],
        'Precision': [],
        'F1-score': [],
        'Sensitivity': [],
        'Specificity': [],
        'Classwise_Accuracy': []
    }
    
    # Compute metrics for each class
    for i, class_name in enumerate(class_names):
        # True Positives (TP): Correct predictions for this class
        TP = cm[i, i]
        
        # False Negatives (FN): Actual class i predicted as something else
        FN = cm[i, :].sum() - TP
        
        # False Positives (FP): Predicted as class i but actually another class
        FP = cm[:, i].sum() - TP
        
        # True Negatives (TN): All other classes correctly not predicted as class i
        TN = cm.sum() - (TP + FP + FN)
        
        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity = recall  # Sensitivity is the same as recall in multi-class
        
        # F1-score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity: TN / (TN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # Classwise Accuracy (CA): (TP + TN) / (TP + TN + FP + FN)
        ca = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        # Append metrics to dictionary
        metrics_dict['Class'].append(class_name)
        metrics_dict['Precision'].append(precision)
        metrics_dict['F1-score'].append(f1_score)
        metrics_dict['Sensitivity'].append(sensitivity)
        metrics_dict['Specificity'].append(specificity)
        metrics_dict['Classwise_Accuracy'].append(ca)
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(os.path.join(output_dir, "classwise_metrics.csv"), index=False)
    
    return metrics_df

def compute_metrics(true_labels, predictions, probabilities, top5_correct, total, class_names, output_dir, features, tsne_labels):
    os.makedirs(output_dir, exist_ok=True)
    
    # Accuracy and BACC
    accuracy = (predictions == true_labels).mean()
    bacc = balanced_accuracy_score(true_labels, predictions)
    top5_accuracy = top5_correct / total
    
    # Compute classwise metrics (precision, f1-score, sensitivity, specificity, CA)
    classwise_metrics_df = compute_classwise_metrics(true_labels, predictions, class_names, output_dir)
    
    # Normalized Confusion Matrix
    cm = confusion_matrix(true_labels, predictions, normalize='true')
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    
    # Normalized Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Classification Report
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    # AUC and ROC Curves
    auc_scores = {}
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_labels == i, probabilities[:, i])
        auc_score = roc_auc_score(true_labels == i, probabilities[:, i])
        auc_scores[class_name] = auc_score
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc_score:.3f})")
    
    # Macro-average AUC
    macro_auc = np.mean(list(auc_scores.values()))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # t-SNE Plot
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    colors = ["#0066CC", '#FF8000', 'darkgreen', '#FF0000', '#800080', '#8B4513']  # Exact hex codes from reference image
    for i, class_name in enumerate(class_names):
        mask = tsne_labels == i
        plt.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], c=colors[i], label=class_name, alpha=0.8)
    
    plt.title("t-SNE Visualization of Test Set Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        "accuracy": accuracy,
        "bacc": bacc,
        "top5_accuracy": top5_accuracy,
        "auc_scores": auc_scores,
        "macro_auc": macro_auc
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    args = parse_args()
    
    # Meta data columns (same as main_pad.py)
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
    
    # Device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = resnet_pad(im_size=224, num_classes=6, attention=True)
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except pickle.UnpicklingError:
        # Fallback for full model object
        model = torch.load(args.model_path, map_location=device, weights_only=False)
    model = model.to(device)
    
    # Load test data
    test_dataset, test_labels = load_test_data(args.data_path, meta_data_columns)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Class names
    class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    
    # Extract features for t-SNE
    features, tsne_labels = extract_features(model, test_loader, device)
    
    # Test model
    predictions, true_labels, probabilities, top5_correct, total = test_model(model, test_loader, device)
    
    # Compute and save metrics
    metrics = compute_metrics(true_labels, predictions, probabilities, top5_correct, total, class_names, args.output_dir, features, tsne_labels)
    
    # Print results
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"BACC: {metrics['bacc']:.3f}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.3f}")
    print(f"Macro AUC: {metrics['macro_auc']:.3f}")
    print("Per-class AUC:")
    for class_name, auc_score in metrics['auc_scores'].items():
        print(f"  {class_name}: {auc_score:.3f}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
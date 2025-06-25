import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score, accuracy_score
import logging

def train(model, train_loader, loss_func, optimizer, is_gpu=True):
    """执行一个训练轮次
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        use_gpu: 是否使用GPU
    Returns:
        model: 更新后的模型
        epoch_loss: 本轮次的总损失
        epoch_accuracy: 本轮次的准确率
        time_elapsed: 训练耗时
    """
    model.train()
    train_loss = 0
    train_acc = 0
    time_start = time.time()
    
    for imgs_batch, metadata_batch, batch_y in train_loader:
        imgs_batch = imgs_batch.float()
        metadata_batch = metadata_batch.float()
        batch_y = batch_y.long()
        
        if is_gpu:
            imgs_batch = imgs_batch.cuda()
            metadata_batch = metadata_batch.cuda()
            batch_y = batch_y.cuda()
        
        optimizer.zero_grad()
        outputs = model(imgs_batch, metadata_batch)
        multi_out = outputs[0]
        
        loss = loss_func(multi_out, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs_batch.size(0)
        _, predicted = torch.max(multi_out, 1)
        train_acc += (predicted == batch_y).sum().item()
    
    time_elapsed = time.time() - time_start
    return model, train_loss, train_acc, time_elapsed

def test(model, test_loader, loss_func, is_gpu=True, top5=False):
    """评估模型性能
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        loss_func: 损失函数
        is_gpu: 是否使用GPU
        top5: 是否计算top5准确率
    Returns:
        test_loss: 测试损失
        test_acc: 准确率
        test_bacc: 平衡准确率
        top5_correct: top5正确数
        time_elapsed: 评估耗时
        predictions: 预测结果
        real_labels: 真实标签
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    test_bacc = 0
    top5_correct = 0
    time_start = time.time()
    predictions = []
    real_labels = []
    
    with torch.no_grad():
        for imgs_batch, metadata_batch, batch_y in test_loader:
            imgs_batch = imgs_batch.float()
            metadata_batch = metadata_batch.float()
            batch_y = batch_y.long()
            
            if is_gpu:
                imgs_batch = imgs_batch.cuda()
                metadata_batch = metadata_batch.cuda()
                batch_y = batch_y.cuda()
            
            outputs = model(imgs_batch, metadata_batch)
            multi_out = outputs[0]
            loss = loss_func(multi_out, batch_y)
            
            test_loss += loss.item() * imgs_batch.size(0)
            _, predicted = torch.max(multi_out, 1)
            test_acc += (predicted == batch_y).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            real_labels.extend(batch_y.cpu().numpy())
            
            if top5:
                _, top5_preds = multi_out.topk(5, 1, True, True)
                top5_correct += top5_preds.eq(batch_y.view(-1, 1).expand_as(top5_preds)).sum().item()
    
    test_bacc = balanced_accuracy_score(real_labels, predictions)
    
    scc_recall = recall_score(real_labels, predictions, labels=[4], average=None, zero_division=0)[0] if 4 in real_labels else 0
    bcc_mask = [i for i, label in enumerate(real_labels) if label == 1]
    bcc_acc = accuracy_score([real_labels[i] for i in bcc_mask], [predictions[i] for i in bcc_mask]) if bcc_mask else 1.0
    
    logging.info(f"SCC 召回率: {scc_recall:.3f}, BCC 准确率: {bcc_acc:.3f}")
    
    time_elapsed = time.time() - time_start
    return test_loss, test_acc, test_bacc, top5_correct, time_elapsed, predictions, real_labels
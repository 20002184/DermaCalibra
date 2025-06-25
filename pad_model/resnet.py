import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

class AMCR(nn.Module):
    def __init__(self, channel=2048, meta_dim=81, reduction=16):
        super(AMCR, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channel),
            nn.Softplus()  # 确保正值不确定性
        )
        self.meta_attention = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, channel),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        # 初始化注意力为近单位变换
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 初始化元数据注意力为低影响
        for m in self.meta_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 初始化不确定性为小正值
        for m in self.uncertainty.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, metadata):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        attn = self.attention(y)  # 通道注意力
        uncertainty = self.uncertainty(metadata)  # 不确定性估计
        meta_attn = self.meta_attention(metadata)  # 元数据注意力
        # 结合：高不确定性通道权重降低
        scaling = attn * meta_attn / (uncertainty + 1e-6)
        scaling = scaling.view(b, c, 1, 1)
        return x * scaling

class resnet_pad(nn.Module):
    def __init__(self, im_size, num_classes, attention=False):
        super(resnet_pad, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.attention = attention
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = True

        self.initial = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
        )
        
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.amcr = AMCR(channel=2048, meta_dim=81)  # 新增AMCR模块
        self.avg_pool = self.model.avgpool
        
        self.metadata_preprocessor = nn.Sequential(
            nn.Linear(81, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

        self.dynamic_conv_generator = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Sigmoid()
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8)

        self.projector1 = nn.Conv2d(256, 2048, kernel_size=1, padding=2, bias=False)
        self.projector2 = nn.Conv2d(512, 2048, kernel_size=1, padding=2, bias=False)
        self.projector3 = nn.Conv2d(1024, 2048, kernel_size=1, padding=2, bias=False)

        self.classifier_block = nn.Sequential(
            nn.Linear(2048 * 4, 2048, bias=False),
            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes, bias=False)
        )

    def forward(self, input_x, metadata):
        input_x = self.initial(input_x)
        l1 = self.layer1(input_x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l4 = self.amcr(l4, metadata)  # 应用AMCR重校准

        metadata_pro = self.metadata_preprocessor(metadata)
        modulation = self.dynamic_conv_generator(metadata_pro)
        l4_dynamic = l4 * modulation.unsqueeze(-1).unsqueeze(-1)
        l4 = l4 + l4_dynamic

        g_vector = self.avg_pool(l4)

        l1 = self.projector1(l1)
        l2 = self.projector2(l2)
        l3 = self.projector3(l3)

        c1, g1 = self.linearAttentionBlock(l1, g_vector)
        c2, g2 = self.linearAttentionBlock(l2, g_vector)
        c3, g3 = self.linearAttentionBlock(l3, g_vector)

        g_vector = g_vector.reshape(g_vector.shape[0], g_vector.shape[1]).unsqueeze(0)
        metadata_pro = metadata_pro.unsqueeze(0)
        global_meta, _ = self.cross_attention(g_vector, metadata_pro, metadata_pro)
        global_meta = global_meta.squeeze(0)

        x = torch.cat((g1, g2, g3, global_meta), dim=1)
        x = self.classifier_block(x)

        return [x, c1, c2, c3]

    def linearAttentionBlock(self, l, g, normlize_method="softmax"):
        N, C, H, W = l.size()
        c = (l * g).sum(dim=1).view(N, 1, H, W)
        if normlize_method == "softmax":
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)
        elif normlize_method == "sigmoid":
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if normlize_method == "softmax":
            g = g.view(N, C, -1).sum(dim=2)
        elif normlize_method == "sigmoid":
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, H, W), g
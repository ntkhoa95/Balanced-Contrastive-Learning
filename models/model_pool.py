import torchvision.models as models
import torch 
import torch.nn as nn
from torch.nn import functional as F
import timm

class ModelwEmb(nn.Module):
    def __init__(
        self,
        num_classes=4,
        arch="convnext",
        pretrained=True,
        print_model=False,
        use_norm=False,
        size=224,
        feat_dim=1024
    ):
        super().__init__()
        self.num_classes=num_classes
        self.arch=arch
        self.pretrained=pretrained
        self.print_model=print_model
        self.use_norm=use_norm
        self.size=size
        self.feat_dim=feat_dim

        self.model = timm.create_model(
            self.arch, pretrained=self.pretrained, num_classes=self.num_classes
        )

        if str(self.arch).startswith("convnext") or str(self.arch).startswith("maxvit"):
            dim_in = self.model.head.fc.in_features
            self.encoder = self.model.forward_features
            self.head_fc = nn.Sequential(
                nn.Linear(dim_in, dim_in), 
                nn.BatchNorm1d(dim_in), 
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, self.feat_dim)
            )
        else:
            # self.encoder = nn.Sequential(*(list(self.model.children())[:-1]))
            raise NotImplementedError(
                'model not supported'
            )

    def forward(self, x):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.model.forward_head(feat, pre_logits=True), dim=1)
        logits = self.model.forward_head(feat)
        fc_w = self.model.get_classifier().weight
        centers_logits = F.normalize(self.head_fc(fc_w), dim=1)
        return feat_mlp, logits, centers_logits

if __name__ == "__main__":
    model = ModelwEmb(
        num_classes=6,
        arch="maxvit_base_tf_224.in21k",
        pretrained=True,
        print_model=False,
        use_norm=False,
        size=224,
        feat_dim=768
    )
    x = torch.randn(16,3,224,224)
    feat_mlp, logits, centers_logits = model(x)
    print(f"feat_mlp: {feat_mlp.shape}, logits: {logits.shape}, centers_logits: {centers_logits.shape}")

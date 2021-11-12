import torch
from torch import nn
from torchvision import models

from .frozen_batchnorm import FrozenBatchNorm2d


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks.
    """
    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, X):
        X = self._feature_extractor(X)
        return X

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class OccNetSimplePointNet(BaseFeatureExtractor):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''
    def __init__(self,
                 freeze_bn,
                 feature_size,
                 c_dim=128,
                 dim=3,
                 hidden_dim=128,
                 depth=4,
                 learnable_attention=False,
                 attention_type='none',
                 attention_dim=750,
                 attention_reduction_type='max'):
        super(OccNetSimplePointNet, self).__init__()
        self._feature_size = feature_size
        c_dim = feature_size
        self.c_dim = c_dim
        self.out_dim = c_dim
        self.depth = depth

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fcs = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(self.depth)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, **kwargs):
        p = p[:, :, :3]
        # p = p.transpose(1, 2)
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)

        for idx, fc in enumerate(self.fcs):
            if idx == len(self.fcs) - 1:
                net = fc(self.actvn(net))
            else:
                net = fc(self.actvn(net))
                pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
                net = torch.cat([net, pooled], dim=2)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))
        return c

class ResNet18(BaseFeatureExtractor):
    """Build a feature extractor using the pretrained ResNet18 architecture for
    image based inputs.
    """
    def __init__(self, freeze_bn, feature_size):
        super(ResNet18, self).__init__()
        self._feature_size = feature_size

        self._feature_extractor = models.resnet18(pretrained=True)
        if freeze_bn:
            FrozenBatchNorm2d.freeze(self._feature_extractor)

        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))


class VoxelEncoder(BaseFeatureExtractor):
    """Simple 3D convolutional encoder network that is used for voxel input.
    Architecture adapted from the Voxel Super-Resolution experiment of the
    "Occupancy Networks: Learning 3D Reconstruction in Function Space"
    """
    def __init__(self, freeze_bn, feature_size):
        super().__init__()
        self._feature_size = feature_size

        self._feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv3d(128, 256, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv3d(256, 512, 3, padding=1, stride=2), nn.ReLU()
        )
        self._feature_extractor_fc = nn.Linear(512 * 2 * 2 * 2, feature_size)

    def forward(self, X):
        X = self._feature_extractor(X)
        return self._feature_extractor_fc(X.view(X.shape[0], -1))


def get_feature_extractor(name, freeze_bn=False, feature_size=128):
    """Based on the name return the appropriate feature extractor."""
    return {
        "resnet18": ResNet18(freeze_bn, feature_size),
        "voxel": VoxelEncoder(freeze_bn, feature_size),
        "simple_pointnet": OccNetSimplePointNet(freeze_bn, feature_size)
    }[name]

# 使用MOCO V3的思想进行训练
# 冻结PatchEmbedding使训练稳定
import torch
import platform

system_type = platform.system()
# 'Windows' 'Linux'
if system_type == 'Windows':
    from BaseModels.VisionTransformer.model_register import *

if system_type == 'Linux':
    import sys

    sys.path.append('/home/ubuntu/workdic/UrbanSceneNet/urban-scene-seg-net/BaseModels/VisionTransformer')
    from model_register import *


class Encoder(nn.Module):
    def __init__(self, backbone=vit_feature_extractor_without_pretrain,
                 img_size=224,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 only_cls=True):
        super(Encoder, self).__init__()
        self.backbone = backbone(img_size=img_size,
                                 patch_size=patch_size,
                                 embed_dim=embed_dim,
                                 depth=depth,
                                 num_heads=num_heads,
                                 only_cls=only_cls)

    def forward(self, x):
        x = self.backbone(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=128):
        super(MLPHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class TinyMoCo(nn.Module):
    def __init__(self, base_encoder=Encoder, dim=128):
        super(TinyMoCo, self).__init__()

        self.encoder = nn.Sequential(
            base_encoder(),
            MLPHead(in_dim=768, out_dim=128)
        )

    def forward(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        # compute query features
        x = self.encoder(x)  # queries: NxC
        x = nn.functional.normalize(x, dim=1)
        labels = torch.zeros(x.shape[0], dtype=torch.long).cuda()
        return x, labels


class MoCo(nn.Module):
    def __init__(self, backbone=Encoder,
                 image_size=224,
                 dim=128,
                 K=16384,
                 m=0.999,
                 T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = nn.Sequential(
            backbone(img_size=image_size, only_cls=True),
            MLPHead(in_dim=768, out_dim=dim)
        )
        self.encoder_k = nn.Sequential(
            backbone(img_size=image_size, only_cls=True),
            MLPHead(in_dim=768, out_dim=dim)
        )

        # initialize
        for param_q, param_k in zip(
                self.encoder_q.parameters(),
                self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, img_q, img_k):
        q = self.encoder_q(img_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(img_k)
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels

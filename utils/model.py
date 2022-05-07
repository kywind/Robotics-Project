import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Config, TransformerEncoderBackbone
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class LocoTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, load=""):
        assert observation_space.shape[0] == 16477
        super(LocoTransformer, self).__init__(observation_space, features_dim)
        self.proprio_embedding = MultiLayerPerceptron(93, 256, 128)
        self.depth_embedding = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1)
        )
        self.transformer_encoder = TransformerEncoderBackbone(Config(max_position_embeddings=100,
                                                                     n_embed=128,
                                                                     n_layer=2,
                                                                     n_head=8,
                                                                     pad_token_id=None,
                                                                     ffn_dim=256))
        self.projection_head = MultiLayerPerceptron(256, 256, features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proprio_input, depth_input_orig = x[:, :93], x[:, 93:].reshape(-1, 4, 64, 64)
        depth_input = 2 * depth_input_orig - 1
        proprio_embed = self.proprio_embedding(proprio_input).unsqueeze(1)
        depth_embed = self.depth_embedding(depth_input).reshape(-1, 128, 16).transpose(1, 2).contiguous()
        cat_embed = torch.cat([proprio_embed, depth_embed], dim=1)
        cat_feature, final_attn_map = self.transformer_encoder(cat_embed) # attn_map: (bsz * num_heads, tgt_len, src_len)
        proprio_feature, depth_feature = cat_feature[:, 0, :], cat_feature[:, 1:, :].mean(dim=1)
        output = self.projection_head(torch.cat([proprio_feature, depth_feature], dim=1))
        # out_dir = 'temp.pth'
        # torch.save(self.state_dict(), out_dir)
        import cv2, random, numpy
        img = depth_input_orig[0,0].cpu().numpy()[:, :, None] * numpy.array([255, 255, 255])
        final_attn_map = final_attn_map.sum(0)[0, 1:].reshape(4, 4).cpu().numpy()
        final_attn_map = numpy.exp(final_attn_map * 100) / numpy.exp(final_attn_map * 100).sum()
        final_attn_map = cv2.resize(final_attn_map, (64, 64))
        final_attn_map = final_attn_map[:, :, None] * numpy.array([0, 0, 255])
        img = img * 0.7 + 0.3 * final_attn_map
        cv2.imwrite('temp/{}.jpg'.format(random.randint(0, 100)), img)
        return output

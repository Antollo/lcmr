import torch
import torch.nn as nn
from torchtyping import TensorType

from lcmr.grammar import Scene
from lcmr.modeler import Modeler
from lcmr.utils.guards import typechecked, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim


@typechecked
class DummyModeler(Modeler):
    def __init__(
        self,
        encoder_feature_dim: int = 2048,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv2d(encoder_feature_dim, hidden_dim, kernel_size=1, padding=1),
            nn.GELU(),
        )
        
        self.skip = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
        )

        self.conv_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

        # prediction heads
        self.to_translation = nn.Linear(hidden_dim, 2)
        self.to_scale = nn.Linear(hidden_dim, 2)
        self.to_color = nn.Linear(hidden_dim, 3)
        self.to_confidence = nn.Linear(hidden_dim, 1)
        self.to_angle = nn.Linear(hidden_dim, 2)

    def forward(self, x: TensorType[batch_dim, reduced_height_dim, reduced_width_dim, channel_dim, torch.float32]) -> Scene:
        
        hidden_state = self.input_projection(x)
        hidden_state = self.conv_net(hidden_state) + self.skip(hidden_state)
        hidden_state = self.linear(hidden_state)
        hidden_state = hidden_state[..., None, :]

        device = next(self.parameters()).device

        translation = torch.sigmoid(self.to_translation(hidden_state)).unsqueeze(1)
        scale = torch.sigmoid(self.to_scale(hidden_state)).unsqueeze(1)
        color = torch.sigmoid(self.to_color(hidden_state)).unsqueeze(1)
        confidence = torch.sigmoid(self.to_confidence(hidden_state)).unsqueeze(1)
        rotation_vec = torch.tanh(self.to_angle(hidden_state))
        rotation_vec = nn.functional.normalize(rotation_vec, dim=-1)
        angle = torch.atan2(rotation_vec[..., 0, None], rotation_vec[..., 1, None]).unsqueeze(1)

        return Scene.from_tensors_sparse(translation=translation, scale=scale, color=color, confidence=confidence, angle=angle, device=device)

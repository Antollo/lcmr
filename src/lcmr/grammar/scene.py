import torch
from torchtyping import TensorType

from lcmr.utils.guards import checked_tensorclass, typechecked, batch_dim, layer_dim, object_dim
from lcmr.grammar.layer import Layer
from lcmr.grammar.object import Object
from lcmr.grammar.transformations import Affine, LazyAffine
from lcmr.grammar.appearance import Appearance


@checked_tensorclass
class Scene:
    layer: Layer

    @staticmethod
    @typechecked
    def from_tensors(
        translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        angle: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
        color: TensorType[batch_dim, layer_dim, object_dim, 3, torch.float32],
        confidence: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
    ) -> "Scene":
        batch_len, layer_len, object_len, _ = translation.shape

        scene = Scene(
            batch_size=[batch_len],
            layer=Layer(
                batch_size=[batch_len, layer_len],
                object=Object(
                    batch_size=[batch_len, layer_len, object_len],
                    objectShape=torch.ones(batch_len, layer_len, object_len, 1, dtype=torch.uint8),
                    transformation=LazyAffine.from_tensors(translation, scale, angle),
                    appearance=Appearance(batch_size=[batch_len, layer_len, object_len], confidence=confidence, color=color),
                ),
                scale=torch.ones(batch_len, layer_len, 1),
                composition=torch.ones(batch_len, layer_len, 1, dtype=torch.uint8),
            ),
        )

        return scene

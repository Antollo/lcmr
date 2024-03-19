import torch
from torchtyping import TensorType
from typing import Optional

from lcmr.utils.guards import checked_tensorclass, typechecked, batch_dim, layer_dim, object_dim, grid_width, grid_height
from lcmr.grammar.layer import Layer
from lcmr.grammar.object import Object
from lcmr.grammar.transformations import LazyAffine
from lcmr.grammar.appearance import Appearance


@checked_tensorclass
class Scene:
    layer: Layer
    backgroundColor: Optional[TensorType[batch_dim, 3, torch.float32]] = None

    @staticmethod
    @typechecked
    def from_tensors_sparse(
        translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        angle: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
        color: TensorType[batch_dim, layer_dim, object_dim, 3, torch.float32],
        confidence: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
        objectShape: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.uint8]] = None,
        fourierCoefficients: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 4, torch.float32]] = None,
        backgroundColor: Optional[TensorType[batch_dim, 3, torch.float32]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> "Scene":
        batch_len, layer_len, object_len, _ = translation.shape

        scene = Scene(
            batch_size=[batch_len],
            layer=Layer(
                batch_size=[batch_len, layer_len],
                object=Object(
                    batch_size=[batch_len, layer_len, object_len],
                    objectShape=objectShape if objectShape != None else torch.ones(batch_len, layer_len, object_len, 1, dtype=torch.uint8, device=device),
                    transformation=LazyAffine.from_tensors(translation, scale, angle),
                    appearance=Appearance(batch_size=[batch_len, layer_len, object_len], confidence=confidence, color=color),
                    fourierCoefficients=fourierCoefficients,
                ),
                scale=torch.ones(batch_len, layer_len, 1, device=device),
                composition=torch.ones(batch_len, layer_len, 1, dtype=torch.uint8, device=device),
            ),
            backgroundColor=backgroundColor,
            device=device,
        )

        return scene

    @staticmethod
    @typechecked
    def from_tensors_dense(
        translation: TensorType[batch_dim, layer_dim, grid_height, grid_width, 2, torch.float32],
        object_scale: TensorType[batch_dim, layer_dim, grid_height, grid_width, 2, torch.float32],
        angle: TensorType[batch_dim, layer_dim, grid_height, grid_width, 1, torch.float32],
        color: TensorType[batch_dim, layer_dim, grid_height, grid_width, 3, torch.float32],
        confidence: TensorType[batch_dim, layer_dim, grid_height, grid_width, 1, torch.float32],
        coordinates_scale: TensorType[batch_dim, layer_dim, 2, torch.float32],
        device: torch.device = torch.device("cpu"),
    ) -> "Scene":
        batch_len, layer_len, grid_height, grid_width, _ = translation.shape
        object_len = grid_height * grid_width

        # scale the translation with the coordinates_scale
        # there is an inner rectangle which determines maximal movement of an object in both x and y directions
        # we rescale the translations which are provided in the inner rectangle to the shape of the outer rectangle.
        inner_square_coords = translation
        translation_inner_rectangle = (torch.tensor([1.0, 1.0], dtype=torch.float32)[None, None, ...] - coordinates_scale) / 2
        scaling_inner_rectangle = coordinates_scale / (torch.tensor([1.0, 1.0], dtype=torch.float32)[None, None, ...] - translation_inner_rectangle)
        inner_square_coords *= scaling_inner_rectangle
        inner_square_coords = inner_square_coords * (1 - translation_inner_rectangle) + translation_inner_rectangle
        translation = inner_square_coords

        # Calculate the size of each grid cell
        grid_size = 1.0 / torch.tensor([grid_width, grid_height], dtype=torch.float32)[None, None, None, None, :]

        # Convert grid coordinates to image coordinates
        translation = translation * grid_size

        # Add the coordinates of the top-left corner of each grid cell to the translation
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.float32) / grid_height, torch.arange(grid_width, dtype=torch.float32) / grid_width, indexing="ij"
        )
        translation_add = torch.stack([grid_x, grid_y], dim=-1)[None, None, :, :, :]
        translation = translation + translation_add

        # scale the object scale with the grid height and width
        object_scale = object_scale * grid_size

        # flatten across object_dim
        translation = translation.reshape(batch_len, layer_len, object_len, 2)
        object_scale = object_scale.reshape(batch_len, layer_len, object_len, 2)
        angle = angle.reshape(batch_len, layer_len, object_len, 1)
        color = color.reshape(batch_len, layer_len, object_len, 3)
        confidence = confidence.reshape(batch_len, layer_len, object_len, 1)

        scene = Scene.from_tensors_sparse(translation, object_scale, angle, color, confidence, device=device)
        return scene

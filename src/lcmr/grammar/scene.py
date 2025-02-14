from typing import Optional, Union

import torch
from torch.nn.functional import binary_cross_entropy, cosine_similarity, l1_loss, mse_loss, normalize
from torchtyping import TensorType

import lcmr.utils.matcher as matcher
from lcmr.grammar.appearance import Appearance
from lcmr.grammar.layer import Layer
from lcmr.grammar.object import Object
from lcmr.grammar.scene_fields import SceneFields
from lcmr.grammar.transformations import LazyAffine
from lcmr.utils.elliptic_fourier_descriptors import normalize_efd, reconstruct_contour, compute_rotational_symmetry
from lcmr.utils.guards import batch_dim, checked_tensorclass, grid_height, grid_width, layer_dim, object_dim, typechecked


@checked_tensorclass
class Scene:
    layer: Layer
    backgroundColor: Optional[TensorType[batch_dim, 3, torch.float32]] = None

    @property
    def fields(self) -> SceneFields:
        return SceneFields(
            translation=self.layer.object.transformation.translation,
            scale=self.layer.object.transformation.scale,
            angle=self.layer.object.transformation.angle,
            rotation_vec=self.layer.object.transformation.rotation_vec,
            color=self.layer.object.appearance.color,
            confidence=self.layer.object.appearance.confidence,
            efd=self.layer.object.efd,
            background_color=self.backgroundColor,
        )

    @staticmethod
    @typechecked
    def from_tensors_sparse(
        translation: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        scale: TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32],
        color: TensorType[batch_dim, layer_dim, object_dim, 3, torch.float32],
        confidence: TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32],
        angle: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.float32]] = None,
        rotation_vec: Optional[TensorType[batch_dim, layer_dim, object_dim, 2, torch.float32]] = None,
        objectShape: Optional[TensorType[batch_dim, layer_dim, object_dim, 1, torch.uint8]] = None,
        efd: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 4, torch.float32]] = None,
        shapeLatent: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, -1, torch.float32]] = None,
        contour: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 2, torch.float32]] = None,
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
                    transformation=LazyAffine.from_tensors(translation=translation, scale=scale, angle=angle, rotation_vec=rotation_vec),
                    appearance=Appearance(batch_size=[batch_len, layer_len, object_len], confidence=confidence, color=color),
                    efd=efd,
                    shapeLatent=shapeLatent,
                    contour=contour,
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
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_height, dtype=torch.float32) / grid_height, torch.arange(grid_width, dtype=torch.float32) / grid_width, indexing="ij")
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

    @staticmethod
    @typechecked
    def rand_like(input: "Scene") -> "Scene":
        # TODO: handle not only LazyAffine?
        # TODO: rand efd?
        scene: Scene = input.clone()
        if type(scene.layer.object.transformation) == LazyAffine:
            torch.rand_like(scene.layer.object.transformation.translation, out=scene.layer.object.transformation.translation)
            torch.rand_like(scene.layer.object.transformation.scale, out=scene.layer.object.transformation.scale)
            torch.rand_like(scene.layer.object.transformation.angle, out=scene.layer.object.transformation.angle)
            scene.layer.object.transformation.angle[:] *= torch.pi * 2

        torch.rand_like(scene.layer.object.appearance.color, out=scene.layer.object.appearance.color)
        torch.rand_like(scene.layer.object.appearance.confidence, out=scene.layer.object.appearance.confidence)
        torch.rand_like(scene.backgroundColor, out=scene.backgroundColor)

        return scene

    @staticmethod
    @typechecked
    def interpolate(start: "Scene", end: "Scene", weight: Union[torch.Tensor, float]) -> "Scene":
        # TODO: handle not only LazyAffine?
        start: Scene = start.clone()
        if type(start.layer.object.transformation) == LazyAffine and type(end.layer.object.transformation) == LazyAffine:
            start.layer.object.transformation.translation.lerp_(end.layer.object.transformation.translation, weight)
            start.layer.object.transformation.scale.lerp_(end.layer.object.transformation.scale, weight)

            if start.layer.object.transformation.angle != None:
                shortest_angle = ((end.layer.object.transformation.angle - start.layer.object.transformation.angle) + torch.pi) % (2 * torch.pi) - torch.pi
                start.layer.object.transformation.angle[:] += weight * shortest_angle

            if start.layer.object.transformation.rotation_vec != None:
                s_vec = start.layer.object.transformation.rotation_vec
                e_vec = end.layer.object.transformation.rotation_vec
                s_vec = normalize(s_vec, dim=-1)
                e_vec = normalize(e_vec, dim=-1)
                start.layer.object.transformation.rotation_vec[:] = normalize(torch.lerp(s_vec, e_vec, weight), dim=-1)

        if start.layer.object.efd != None and end.layer.object.efd != None:
            # TODO: Does it have sense? Not really...
            start.layer.object.efd.lerp_(end.layer.object.efd, weight)
            start.layer.object.efd[:] = normalize_efd(start.layer.object.efd)

        start.layer.object.appearance.color.lerp_(end.layer.object.appearance.color, weight)
        start.layer.object.appearance.confidence.lerp_(end.layer.object.appearance.confidence, weight)
        start.backgroundColor.lerp_(end.backgroundColor, weight)

        return start

    @staticmethod
    @typechecked
    def dist(scene_true: "Scene", scene_pred: "Scene", fields: str = "tsrceb", aggregate: bool = True) -> Union[torch.Tensor, list[torch.Tensor]]:
        a = scene_true
        b = scene_pred

        object_len_a = a.layer.object.shape[-1]
        object_len_b = b.layer.object.shape[-1]
        assert object_len_b >= object_len_a

        fields_a = a.fields
        fields_b = b.fields

        ind_a, ind_b, confidence_b = matcher.match(
            (1.0, fields_a.translation, fields_b.translation),
            (0.1, fields_a.color, fields_b.color),
            (0.05, fields_a.confidence, fields_b.confidence),
        )

        # TODO: do not copy, at least do not deep copy
        a: Scene = a.clone()
        b: Scene = b.clone()

        a.layer.object = matcher.gather(ind_a, a.layer.object)
        b.layer.object = matcher.gather(ind_b, b.layer.object)
        fields_a = a.fields
        fields_b = b.fields

        total = []

        assert torch.isfinite(fields_b.confidence).all()
        assert torch.isfinite(confidence_b).all()

        total.append(binary_cross_entropy(fields_b.confidence.clip(0, 1), confidence_b.clip(0, 1)))

        for field_name in fields:
            field_a = fields_a[field_name]
            field_b = fields_b[field_name]

            if len(field_a.shape) > 2:
                n = field_a.shape[2]
                field_b = field_b[:, :, :n]

            if field_name == "a":
                dist = 1 - torch.cos(field_a - field_b)
            elif field_name == "r":
                # with torch.no_grad():
                #     sym = compute_rotational_symmetry(fields_a["e"].flatten(0, 2)[..., :5, :]).to(torch.float32)
                # angle_a = torch.atan2(field_a[..., 0, None], field_a[..., 1, None]).flatten(0, 2)
                # angle_b = torch.atan2(field_b[..., 0, None], field_b[..., 1, None]).flatten(0, 2)
                # dist = (1 - torch.cos((angle_a - angle_b) * sym)) ** 2
                dist = (1 - cosine_similarity(field_a, field_b, dim=-1)) ** 2
            elif field_name == "e":
                # dist = l1_loss(field_a, field_b, reduction="none")
                dist = mse_loss(reconstruct_contour(field_a), reconstruct_contour(field_b), reduction="none")
            else:
                dist = mse_loss(field_a, field_b, reduction="none")

            dist = dist.flatten(1).mean()

            if field_name == "t":
                dist = dist * 5
            elif field_name == "r":
                dist = dist * 0.05

            total.append(dist)

        if aggregate:
            total = sum(total)
        return total

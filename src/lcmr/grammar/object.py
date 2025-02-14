from typing import Optional, Union

import torch
from torchtyping import TensorType

from lcmr.grammar.appearance import Appearance
from lcmr.grammar.transformations import Affine, Identity, LazyAffine, Translation
from lcmr.utils.elliptic_fourier_descriptors import reconstruct_contour
from lcmr.utils.guards import batch_dim, checked_tensorclass, layer_dim, object_dim


@checked_tensorclass
class Object:
    objectShape: TensorType[batch_dim, layer_dim, object_dim, 1, torch.uint8]  # Value from Shape2D enum
    transformation: Union[Identity, Translation, Affine, LazyAffine]
    appearance: Appearance
    efd: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 4, torch.float32]] = None
    fd: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 2, torch.float32]] = None
    shapeLatent: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, -1, torch.float32]] = None
    contour: Optional[TensorType[batch_dim, layer_dim, object_dim, -1, 2, torch.float32]] = None

    _n_points: int = -1

    def reconstruct_contour(self, n_points: int):
        if self.contour == None or self._n_points != n_points:
            self._n_points = n_points
            self.contour = reconstruct_contour(self.efd, n_points)
        return self.contour

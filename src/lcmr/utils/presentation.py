import torch
import numpy as np
from torchtyping import TensorType
from PIL import Image
from IPython.display import display, HTML
import sys

from lcmr.utils.guards import typechecked, height_dim, width_dim, channel_dim

in_colab = "google.colab" in sys.modules


@typechecked
def tensor_to_img(img: TensorType[height_dim, width_dim, channel_dim, torch.float32]) -> Image:
    return Image.fromarray((img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


@typechecked
def display_img(img: TensorType[height_dim, width_dim, channel_dim, torch.float32]):
    if in_colab:
        display(
            HTML(
                """<style>
                .output_image>img {
                    background-color: transparent !important;
                }
                </style>"""
            )
        )

    display(tensor_to_img(img).convert("RGB"))

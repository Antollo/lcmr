import torch
import numpy as np
from torchvision.utils import make_grid
from torchtyping import TensorType
from PIL import Image
from IPython.display import display, HTML
from collections.abc import Sequence
import sys
import os

from lcmr.utils.guards import typechecked, ImageBHWC, ImageHWC, channel_dim

in_colab = "google.colab" in sys.modules
in_vscode = "VSCODE_CWD" in os.environ

if in_vscode:
    display(
        HTML(
            """
            <style>
            .cell-output-ipywidget-background {
                background-color: transparent !important;
            }
            :root {
                --jp-widgets-color: var(--vscode-editor-foreground);
                --jp-widgets-font-size: var(--vscode-editor-font-size);
            }  
            </style>
            """
        )
    )


@typechecked
def make_img_grid(imgs: Sequence[ImageBHWC], padding: int = 2, pad_value: int = 1, nrow: int = 8) -> TensorType[-1, -1, channel_dim, torch.float32]:
    return make_grid(torch.cat(imgs, dim=0).permute(0, 3, 1, 2), padding=padding, pad_value=pad_value, nrow=nrow).permute(1, 2, 0)


@typechecked
def tensor_to_img(img: ImageHWC) -> Image:
    return Image.fromarray((img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


@typechecked
def display_img(img: ImageHWC):
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

    display(tensor_to_img(img).convert("RGBA"))

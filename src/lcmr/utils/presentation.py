import os
import sys
from collections.abc import Sequence
from typing import Union

import numpy as np
import torch
from IPython.display import HTML, display
from PIL import Image
from torchtyping import TensorType
from torchvision.utils import make_grid

from lcmr.utils.guards import ImageBHWC, ImageHWC, channel_dim, typechecked

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
def make_img_grid(imgs: Union[Sequence[ImageBHWC], ImageBHWC], padding: int = 2, pad_value: int = 1, nrow: int = 8) -> TensorType[-1, -1, -1, torch.float32]:
    if not torch.is_tensor(imgs):
        imgs = torch.cat(imgs, dim=0)
    return make_grid(imgs.permute(0, 3, 1, 2), padding=padding, pad_value=pad_value, nrow=nrow).permute(1, 2, 0)


@typechecked
def tensor_to_img(img: ImageHWC) -> Image:
    return Image.fromarray((img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


@typechecked
def imgs_to_gif(imgs: Sequence[ImageHWC], name: str, duration: int = 200) -> None:
    imgs = [tensor_to_img(img) for img in imgs]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


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

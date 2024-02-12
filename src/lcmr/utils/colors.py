import torch
from collections import namedtuple
from matplotlib import colors as mcolors

Colors = namedtuple("Colors", mcolors.CSS4_COLORS.keys())
colors = Colors(**{key: torch.tensor(mcolors.to_rgba(value), dtype=torch.float32) for key, value in mcolors.CSS4_COLORS.items()})
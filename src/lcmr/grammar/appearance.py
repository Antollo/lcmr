from dataclasses import dataclass


@dataclass
class Appearance:
    confidence: float
    r: float
    g: float
    b: float

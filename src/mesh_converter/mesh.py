from dataclasses import dataclass
from enum import Enum
from typing import Dict
import numpy as np
import numpy.typing as npt

__all__ = ["Mesh", "CellType", "cell_to_facet"]


class CellType(Enum):
    point = 1
    triangle = 2
    quad = 3
    tetra = 4
    hex = 5
    interval = 6

    @classmethod
    def from_value(cls, value: str):
        """
        Workaround for string enum prior to Python 3.11
        """
        lower = value.lower()
        if lower == "point":
            return cls.point
        elif lower == "triangle":
            return cls.triangle
        elif lower == "quad":
            return cls.quad
        elif lower == "tetra":
            return cls.tetra
        elif lower == "hex":
            return cls.hex
        elif lower == "interval":
            return cls.interval
        else:
            raise ValueError(f"Unknown cell type: {value}")

    def __str__(self) -> str:
        if self == CellType.point:
            return "point"
        elif self == CellType.triangle:
            return "triangle"
        elif self == CellType.quad:
            return "quad"
        elif self == CellType.tetra:
            return "tetra"
        elif self == CellType.hex:
            return "hex"
        elif self == CellType.interval:
            return "interval"
        else:
            raise ValueError(f"Unknown cell type: {self}")

    @property
    def tdim(self):
        if self == CellType.point:
            return 0
        elif self == CellType.interval:
            return 1
        elif self == CellType.triangle:
            return 2
        elif self == CellType.quad:
            return 2
        elif self == CellType.tetra:
            return 3
        elif self == CellType.hex:
            return 3
        else:
            raise ValueError(f"Unknown cell type: {self}")


@dataclass
class Mesh:
    geometry: npt.NDArray[np.floating]  # Mesh nodes
    topology: npt.NDArray[np.int64]  # Connectivity for cells in geometry
    cell_type: CellType  # Cell type
    cell_values: npt.NDArray[np.int64]
    facet_topology: npt.NDArray[np.int64]
    facet_values: npt.NDArray[np.int64]


cell_to_facet: Dict[CellType, CellType] = {
    CellType.triangle: CellType.interval,
    CellType.quad: CellType.interval,
    CellType.tetra: CellType.triangle,
    CellType.hex: CellType.quad,
    CellType.interval: CellType.point,
}

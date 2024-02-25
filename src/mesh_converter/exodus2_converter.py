from pathlib import Path

import netCDF4
import numpy as np
from enum import Enum

from .mesh import Mesh, CellType

# Based on: https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf
tetra_facet_to_vertex_map = {0: [0, 1, 3],
                             1: [1, 2, 3], 2: [0, 2, 3], 3: [0, 1, 2]}
triangle_to_vertex_map = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
quad_to_vertex_map = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]}
hex_to_vertex_map = {0: [0, 1, 4, 5], 1: [1, 2, 5, 6], 2: [2, 3, 6, 7],
                     3: [0, 3, 4, 7], 4: [0, 1, 2, 3], 5: [4, 5, 6, 7]}

side_set_to_vertex_map = {CellType.quad: quad_to_vertex_map,
                          CellType.triangle: triangle_to_vertex_map,
                          CellType.tetra: tetra_facet_to_vertex_map,
                          CellType.hex: hex_to_vertex_map}


class ExodusCellType(Enum):
    TETRA = 1
    HEX = 2
    QUAD = 3
    TRIANGLE = 4

    @classmethod
    def from_value(cls, value: str):
        """
        Workaround for string enum prior to Python 3.11
        """
        upper = value.upper()
        if upper == "TRIANGLE":
            return cls.TRIANGLE
        elif upper == "QUAD":
            return cls.QUAD
        elif upper == "TETRA":
            return cls.TETRA
        elif upper == "HEX":
            return cls.HEX
        else:
            raise ValueError(f"Unknown cell type: {value}")

    def __str__(self) -> str:
        if self == ExodusCellType.TETRA:
            return "tetra"
        elif self == ExodusCellType.HEX:
            return "hex"
        elif self == ExodusCellType.QUAD:
            return "quad"
        elif self == ExodusCellType.TRIANGLE:
            return "triangle"
        else:
            raise ValueError(f"Unknown cell type: {self}")


def read_exodus2_data(filename: str | Path) -> Mesh:
    """
    Read mesh data from an exodus2 file.
    Includes geometry, topology, cell type, and facet data.
    """
    infile = netCDF4.Dataset(filename)

    # use page 171 of manual to extract data
    num_nodes = infile.dimensions["num_nodes"].size
    gdim = infile.dimensions["num_dim"].size
    num_blocks = infile.dimensions["num_el_blk"].size
    assert num_blocks == 1, "Currently only supports single block"

    # Get coordinates of mesh
    coordinates = infile.variables.get("coord")
    if coordinates is None:
        coordinates = np.zeros((num_nodes, gdim), dtype=np.float64)
        for i, coord in enumerate(["x", "y", "z"]):
            coord_i = infile.variables.get(f"coord{coord}")
            if coord_i is not None:
                coordinates[:coord_i.size, i] = coord_i[:]

    # Get element connectivity
    connectivity = infile.variables.get("connect1")
    cell_type = CellType.from_value(
        str(ExodusCellType.from_value(connectivity.elem_type)))
    assert connectivity is not None, "No connectivity found"
    connectivity_array = connectivity[:] - 1

    # For each facet set get the values and the corresponding node indices
    local_facet_index = side_set_to_vertex_map[cell_type]
    num_facet_sets = infile.dimensions["num_side_sets"].size
    values = infile.variables.get("ss_prop1")
    facet_indices = []
    facet_values = []
    for i in range(1, num_facet_sets+1):
        value = values[i-1]
        elements = infile.variables[f"elem_ss{i}"]
        local_facets = infile.variables[f"side_ss{i}"]
        for element, index in zip(elements, local_facets):
            facet_indices.append(
                connectivity_array[element-1, local_facet_index[index-1]])
            facet_values.append(value)
    sub_geometry = np.vstack(facet_indices)
    infile.close()
    return Mesh(geometry=coordinates,
                topology=connectivity_array.astype(np.int64),
                cell_type=cell_type,
                facet_topology=sub_geometry.astype(np.int64),
                facet_values=np.asarray(facet_values))

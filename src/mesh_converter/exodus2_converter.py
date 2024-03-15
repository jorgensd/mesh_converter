from pathlib import Path

import netCDF4
import numpy as np
from enum import Enum

from .mesh import Mesh, CellType

# Based on: https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf
tetra_facet_to_vertex_map = {0: [0, 1, 3], 1: [1, 2, 3], 2: [0, 2, 3], 3: [0, 1, 2]}
# https://coreform.com/cubit_help/appendix/element_numbering.htm
# Note that triangular side-sets goes from 2 to 4 (with 0 base index)
triangle_to_vertex_map = {2: [0, 1], 3: [1, 2], 4: [2, 0]}
quad_to_vertex_map = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]}
hex_to_vertex_map = {
    0: [0, 1, 4, 5],
    1: [1, 2, 5, 6],
    2: [2, 3, 6, 7],
    3: [0, 3, 4, 7],
    4: [0, 1, 2, 3],
    5: [4, 5, 6, 7],
}

side_set_to_vertex_map = {
    CellType.quad: quad_to_vertex_map,
    CellType.triangle: triangle_to_vertex_map,
    CellType.tetra: tetra_facet_to_vertex_map,
    CellType.hex: hex_to_vertex_map,
}


class ExodusCellType(Enum):
    TETRA = 1
    HEX = 2
    QUAD = 3
    TRIANGLE = 4
    INTERVAL = 5

    @classmethod
    def from_value(cls, value: str):
        """
        Workaround for string enum prior to Python 3.11
        """
        upper = value.upper()
        if upper == "TRI3":
            return cls.TRIANGLE
        elif upper == "QUAD":
            return cls.QUAD
        elif upper == "TETRA":
            return cls.TETRA
        elif upper == "HEX":
            return cls.HEX
        elif upper == "BEAM2":
            return cls.INTERVAL
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
        elif self == ExodusCellType.INTERVAL:
            return "interval"
        else:
            raise ValueError(f"Unknown cell type: {self}")


def read_exodus2_data(filename: str | Path) -> Mesh:
    """
    Read mesh data from an exodus2 file.
    Includes geometry, topology, cell type, and facet data.
    """
    try:
        infile = netCDF4.Dataset(filename)

        # use page 171 of manual to extract data
        num_nodes = infile.dimensions["num_nodes"].size
        gdim = infile.dimensions["num_dim"].size
        num_blocks = infile.dimensions["num_el_blk"].size

        # Get coordinates of mesh
        coordinates = infile.variables.get("coord")
        if coordinates is None:
            coordinates = np.zeros((num_nodes, gdim), dtype=np.float64)
            for i, coord in enumerate(["x", "y", "z"]):
                coord_i = infile.variables.get(f"coord{coord}")
                if coord_i is not None:
                    coordinates[: coord_i.size, i] = coord_i[:]

        # Get element connectivity
        connectivity_arrays = []
        cell_types = np.empty(num_blocks, dtype=CellType)
        num_cells_per_block = np.zeros(num_blocks, dtype=np.int32)
        # Create map from topological dimension to block indices
        tdim_to_cell_index = {0: [], 1: [], 2: [], 3: []}
        for i in range(1, num_blocks + 1):
            connectivity = infile.variables.get(f"connect{i}")
            cell_type = CellType.from_value(
                str(ExodusCellType.from_value(connectivity.elem_type))
            )
            cell_types[i - 1] = cell_type
            tdim_to_cell_index[cell_type.tdim].append(i - 1)
            assert connectivity is not None, "No connectivity found"
            connectivity_arrays.append(connectivity[:] - 1)
            num_cells_per_block[i - 1] = connectivity.shape[0]
        max_dim = 0
        for i in range(4):
            tdim_to_cell_index[i] = np.asarray(tdim_to_cell_index[i], dtype=np.int32)
            if len(tdim_to_cell_index[i]) > 0:
                max_dim = i
        cell_block_indices = tdim_to_cell_index[max_dim]
        for cell in cell_types[cell_block_indices]:
            assert (
                cell_types[cell_block_indices[0]] == cell
            ), "Mixed cell types not supported"
        cell_type = cell_types[cell_block_indices][0]
        connectivity_array = np.vstack(
            [connectivity_arrays[i] for i in cell_block_indices]
        )
        # If blocks are used for facet markers
        if "eb_prop1" in infile.variables.keys():
            block_values = infile.variables["eb_prop1"][:]

            # Extract cell block values
            cell_array = np.zeros(connectivity_array.shape[0], dtype=np.int64)
            insert_offset = np.zeros(len(cell_block_indices) + 1, dtype=np.int64)
            insert_offset[1:] = np.cumsum(num_cells_per_block[cell_block_indices])
            for i, index in enumerate(cell_block_indices):
                cell_array[insert_offset[i] : insert_offset[i + 1]] = block_values[
                    index
                ]

            # Get all facet blocks
            facet_blocks_indices = tdim_to_cell_index[max_dim - 1]
            if len(facet_blocks_indices) > 0:
                sub_geometry = np.vstack(
                    [connectivity_arrays[i] for i in facet_blocks_indices]
                )
                facet_values = np.zeros(sub_geometry.shape[0], dtype=np.int64)
                insert_offset = np.zeros(len(facet_blocks_indices) + 1, dtype=np.int64)
                insert_offset[1:] = np.cumsum(num_cells_per_block[facet_blocks_indices])
                for i, index in enumerate(facet_blocks_indices):
                    facet_values[insert_offset[i] : insert_offset[i + 1]] = (
                        block_values[index]
                    )
            # If sidesets are used for facet markers
            elif "ss_prop1" in infile.variables.keys():
                # Extract facet values
                local_facet_index = side_set_to_vertex_map[cell_type]
                if "num_side_sets" not in infile.dimensions:
                    num_vertices_per_facet = len(local_facet_index[0])
                    sub_geometry = np.zeros((0, num_vertices_per_facet), dtype=np.int64)
                    facet_values = np.zeros(0, dtype=np.int64)
                else:
                    infile.dimensions.get("num_side_sets", 0)
                    num_facet_sets = infile.dimensions["num_side_sets"].size
                    values = infile.variables.get("ss_prop1")
                    facet_indices = []
                    facet_values = []
                    for i in range(1, num_facet_sets + 1):
                        value = values[i - 1]
                        elements = infile.variables[f"elem_ss{i}"]
                        local_facets = infile.variables[f"side_ss{i}"]
                        for element, index in zip(elements, local_facets):
                            facet_indices.append(
                                connectivity_array[
                                    element - 1, local_facet_index[index - 1]
                                ]
                            )
                            facet_values.append(value)
                    sub_geometry = np.vstack(facet_indices)
            else:
                sub_geometry = np.zeros((0, 0), dtype=np.int64)
                facet_values = np.zeros(0, dtype=np.int64)

            # Remove duplicate facets
            sub_geometry, indices = np.unique(sub_geometry, axis=0, return_index=True)
            facet_values = np.array(facet_values)[indices]
            if len(facet_values) > 0:
                facet_values = np.hstack(facet_values)

        else:
            cell_array = np.zeros(0, dtype=np.int64)
            sub_geometry = np.zeros((0, 0), dtype=np.int64)
            facet_values = np.zeros(0, dtype=np.int64)
    finally:
        infile.close()
    return Mesh(
        geometry=coordinates,
        topology=connectivity_array.astype(np.int64),
        cell_type=cell_type,
        cell_values=cell_array,
        facet_topology=sub_geometry.astype(np.int64),
        facet_values=facet_values,
    )

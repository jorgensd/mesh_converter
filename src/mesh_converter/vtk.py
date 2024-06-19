from pathlib import Path
from mpi4py import MPI
import h5py
import numpy as np
from mesh_converter import Mesh, CellType
from enum import Enum

__all__ = ["write", "VTKCellType"]


class VTKCellType(Enum):
    """
    VTK Cell types (for arbitrary order Lagrange):
    https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    """

    vertex = 2
    line = 68
    triangle = 69
    quadrilateral = 70
    tetrahedron = 78
    hexahedron = 72

    @classmethod
    def from_value(cls, value: CellType):
        """
        Workaround for string enum prior to Python 3.11
        """
        if value == CellType.point:
            return cls.vertex
        elif value == CellType.triangle:
            return cls.triangle
        elif value == CellType.quad:
            return cls.quadrilateral
        elif value == CellType.tetra:
            return cls.tetrahedron
        elif value == CellType.hex:
            return cls.hexahedron
        elif value == CellType.interval:
            return cls.line
        else:
            raise ValueError(f"Unknown cell type: {value}")

    def __str__(self) -> str:
        if self == VTKCellType.line:
            return "line"
        elif self == VTKCellType.triangle:
            return "triangle"
        elif self == VTKCellType.quadrilateral:
            return "quadrilateral"
        elif self == VTKCellType.tetrahedron:
            return "tetrahedron"
        elif self == VTKCellType.hexahedron:
            return "hexahedron"
        elif self == VTKCellType.vertex:
            return "vertex"
        else:
            raise ValueError(f"Unknown cell type: {self}")

    def __int__(self) -> int:
        return self.value


def write(mesh: Mesh, filename: str | Path):
    """
    Write mesh to VTK HDF5 format
    """
    comm = MPI.COMM_WORLD
    assert comm.size == 1, "Only serial writing supported"
    fname = Path(filename).with_suffix(".vtkhdf")
    inf = h5py.File(fname, "w", driver="mpio", comm=comm)

    metadata = inf.create_group(np.string_("Metadata"))
    metadata.attrs["gdim"] = mesh.geometry.shape[1]
    hdf = inf.create_group(np.string_("VTKHDF"))
    h5py.string_dtype(encoding="ascii")
    hdf.attrs["Version"] = [2, 2]
    hdf.attrs["Type"] = np.string_("UnstructuredGrid")
    global_shape = mesh.geometry.shape
    gdtype = mesh.geometry.dtype

    padded_geometry = np.zeros((global_shape[0], 3), dtype=gdtype)
    padded_geometry[:, : global_shape[1]] = mesh.geometry
    p_string = np.string_("Points")
    geom_set = hdf.create_dataset(p_string, padded_geometry.shape, dtype=gdtype)
    geom_set[:, :] = padded_geometry

    # Put global topology
    top_set = hdf.create_dataset(
        "Connectivity", (mesh.topology_offset[-1],), dtype=np.int64
    )
    top_set[:] = mesh.topology_array

    # Put cell type
    num_cells = len(mesh.topology_offset) - 1
    type_set = hdf.create_dataset("Types", (num_cells,), dtype=np.uint8)

    cts = np.asarray(
        [int(VTKCellType.from_value(ct)) for ct in mesh.cell_types], dtype=np.uint8
    )
    type_set[:] = cts

    # Geom dofmap offset
    con_part = hdf.create_dataset("NumberOfConnectivityIds", (1,), dtype=np.int64)
    con_part[0] = mesh.topology_offset[-1]

    # Num cells
    hdf.create_dataset(
        "NumberOfCells",
        (1,),
        dtype=np.int64,
        data=np.array([num_cells], dtype=np.int64),
    )

    # num points
    num_points = hdf.create_dataset("NumberOfPoints", (1,), dtype=np.int64)
    num_points[0] = mesh.geometry.shape[0]

    # Offsets
    offsets = hdf.create_dataset("Offsets", (num_cells + 1,), dtype=np.int64)
    offsets[:] = mesh.topology_offset

    # Add celldata
    if len(mesh.cell_values) > 0:
        cv = hdf.create_group("CellData")
        cv.attrs["Scalars"] = ["Cell_Markers"]
        cv.create_dataset("Cell_Markers", shape=(num_cells, ), data=mesh.cell_values)

    inf.close()

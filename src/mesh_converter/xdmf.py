from mpi4py import MPI

import adios2
from pathlib import Path
import xml.etree.ElementTree as ET
from enum import Enum
from .mesh import Mesh, CellType, cell_to_facet
import numpy as np
import numpy.typing as npt

__all__ = ["write", "XDMFCellType"]

def resolve_adios_scope(adios2):
    return adios2.bindings if hasattr(adios2, "bindings") else adios2


adios2 = resolve_adios_scope(adios2)


class XDMFCellType(Enum):
    Polyvertex = 1
    Polyline = 2
    Triangle = 3
    Quadrilateral = 4
    Tetrahedron = 5
    Hexahedron = 6

    @classmethod
    def from_value(cls, value: CellType):
        """
        Workaround for string enum prior to Python 3.11
        """
        if value == CellType.point:
            return cls.Polyvertex
        elif value == CellType.triangle:
            return cls.Triangle
        elif value == CellType.quad:
            return cls.Quadrilateral
        elif value == CellType.tetra:
            return cls.Tetrahedron
        elif value == CellType.hex:
            return cls.Hexahedron
        elif value == CellType.interval:
            return cls.Polyline
        else:
            raise ValueError(f"Unknown cell type: {value}")

    def __str__(self) -> str:
        if self == XDMFCellType.Polyline:
            return "Polyline"
        elif self == XDMFCellType.Triangle:
            return "Triangle"
        elif self == XDMFCellType.Quadrilateral:
            return "Quadrilateral"
        elif self == XDMFCellType.Tetrahedron:
            return "Tetrahedron"
        elif self == XDMFCellType.Hexahedron:
            return "Hexahedron"
        elif self == XDMFCellType.Polyvertex:
            return "Polyvertex"
        else:
            raise ValueError(f"Unknown cell type: {self}")


def extract_shape(topology_offset: npt.NDArray)->tuple[int, int]:
    """
    Extract topology shape for single cell mesh    
    """
    if len(topology_offset) == 1:
        return (0, 0)
    num_nodes_per_cell = set(topology_offset[1:] - topology_offset[:-1])
    assert len(num_nodes_per_cell) == 1, "Mixed meshes not supported"
    return (len(topology_offset)-1, next(iter(num_nodes_per_cell)))

def define_topology(
    topology_offset: npt.NDArray[np.int64],
    cell_type: CellType,
    mesh_element: ET.Element,
    filename: Path,
):  
    topology_shape = extract_shape(topology_offset)
    topology_el = ET.SubElement(mesh_element, "Topology")
    topology_el.attrib["NumberOfElements"] = str(topology_shape[0])
    topology_el.attrib["TopologyType"] = str(XDMFCellType.from_value(cell_type))
    topology_el.attrib["NodesPerElement"] = str(topology_shape[1])
    it0 = ET.SubElement(topology_el, "DataItem")
    it0.attrib["Dimensions"] = f"{topology_shape[0]} {topology_shape[1]}"
    it0.attrib["Format"] = "HDF"
    it0.text = f"{filename.stem}.h5:/Step0/Connectivity_{str(cell_type)}"


def write(mesh: Mesh, filename: str | Path):
    filename = Path(filename)

    xdmf = ET.Element("Xdmf")
    xdmf.attrib["Version"] = "3.0"
    xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
    domain = ET.SubElement(xdmf, "Domain")

    # Define mesh topology
    grid = ET.SubElement(domain, "Grid")
    grid.attrib["GridType"] = "Uniform"
    grid.attrib["Name"] = "Mesh"
    assert len(set(mesh.cell_types)) == 1, "Mixed meshes not supported"
    define_topology(mesh.topology_offset, mesh.cell_types[0], grid, filename)

    # Define mesh geometry
    geometry = ET.SubElement(grid, "Geometry")
    geometry.attrib["GeometryType"] = "XY" if mesh.geometry.shape[1] == 2 else "XYZ"
    it0 = ET.SubElement(geometry, "DataItem")
    it0.attrib["Dimensions"] = f"{mesh.geometry.shape[0]} {mesh.geometry.shape[1]}"
    it0.attrib["Format"] = "HDF"
    it0.text = f"{filename.stem}.h5:/Step0/Points"

    # Add cell values
    if len(mesh.cell_values) > 0:
        attrib = ET.SubElement(grid, "Attribute")
        attrib.attrib["Name"] = "Cell markers"
        attrib.attrib["AttributeType"] = "Scalar"
        attrib.attrib["Center"] = "Cell"
        it1 = ET.SubElement(attrib, "DataItem")
        it1.attrib["Dimensions"] = f"{len(mesh.cell_values)}"
        it1.attrib["Format"] = "HDF"
        it1.attrib["DataType"] = "Int"
        it1.text = f"{filename.stem}.h5:/Step0/Cell_Markers"

    # Define facet topology and geometry
    if len(mesh.facet_values) > 0:
        facet_grid = ET.SubElement(domain, "Grid")
        facet_grid.attrib["GridType"] = "Uniform"
        facet_grid.attrib["Name"] = "Facet_Mesh"
        define_topology(
            mesh.facet_topology_offset, cell_to_facet[mesh.cell_types[0]], facet_grid, filename
        )
        facet_geometry = ET.SubElement(facet_grid, "Geometry")
        facet_geometry.attrib["GeometryType"] = (
            "XY" if mesh.geometry.shape[1] == 2 else "XYZ"
        )
        it0 = ET.SubElement(facet_geometry, "DataItem")
        it0.attrib["Dimensions"] = f"{mesh.geometry.shape[0]} {mesh.geometry.shape[1]}"
        it0.attrib["Format"] = "HDF"
        it0.text = f"{filename.stem}.h5:/Step0/Points"

        # Add facet values
        attrib = ET.SubElement(facet_grid, "Attribute")
        attrib.attrib["Name"] = "Facet markers"
        attrib.attrib["AttributeType"] = "Scalar"
        attrib.attrib["Center"] = "Cell"
        it1 = ET.SubElement(attrib, "DataItem")
        it1.attrib["Dimensions"] = f"{len(mesh.facet_values)}"
        it1.attrib["Format"] = "HDF"
        it1.attrib["DataType"] = "Int"
        it1.text = f"{filename.stem}.h5:/Step0/Facet_Markers"

    with open(filename, "w") as outfile:
        outfile.write(
            '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        outfile.write(ET.tostring(xdmf, encoding="unicode"))

    # Create ADIOS2 writer
    assert MPI.COMM_WORLD.size == 1, "Mesh convert only works in serial for now"
    adios = adios2.ADIOS(MPI.COMM_WORLD)
    io = adios.DeclareIO("Mesh writer")
    io.SetEngine("HDF5")
    outfile = io.Open(str(filename.with_suffix(".h5")), adios2.Mode.Write)
    pointvar = io.DefineVariable(
        "Points",
        mesh.geometry,
        shape=[mesh.geometry.shape[0], mesh.geometry.shape[1]],
        start=[0, 0],
        count=[mesh.geometry.shape[0], mesh.geometry.shape[1]],
    )
    outfile.Put(pointvar, mesh.geometry)
    top_shape = extract_shape(mesh.topology_offset)
    top_data = mesh.topology_array.reshape(*top_shape)
    topology_var = io.DefineVariable(
        f"Connectivity_{str(mesh.cell_types[0])}",
        top_data,
        shape=[top_shape[0], top_shape[1]],
        start=[0, 0],
        count=[top_shape[0], top_shape[1]],
    )
    outfile.Put(topology_var, top_data)

    facet_top_shape = extract_shape(mesh.facet_topology_offset)
    facet_top_data = mesh.facet_topology_array.reshape(*facet_top_shape)
    facet_topology_var = io.DefineVariable(
        f"Connectivity_{str(cell_to_facet[mesh.cell_types[0]])}",
        facet_top_data,
        shape=[facet_top_shape[0], facet_top_shape[1]],
        start=[0, 0],
        count=[facet_top_shape[0], facet_top_shape[1]],
    )
    outfile.Put(facet_topology_var, facet_top_data)

    if len(mesh.cell_values) > 0:
        cell_values_var = io.DefineVariable(
            "Cell_Markers",
            mesh.cell_values,
            shape=[mesh.cell_values.shape[0]],
            start=[0],
            count=[mesh.cell_values.shape[0]],
        )
        outfile.Put(cell_values_var, mesh.cell_values)

    if len(mesh.facet_values) > 0:
        facet_values_var = io.DefineVariable(
            "Facet_Markers",
            mesh.facet_values,
            shape=[mesh.facet_values.shape[0]],
            start=[0],
            count=[mesh.facet_values.shape[0]],
        )
        outfile.Put(facet_values_var, mesh.facet_values)

    outfile.PerformPuts()
    outfile.Close()
    assert adios.RemoveIO("Mesh writer")

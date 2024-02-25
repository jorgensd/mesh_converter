from mpi4py import MPI

import adios2
from pathlib import Path
import xml.etree.ElementTree as ET
from enum import Enum
from .mesh import Mesh, CellType, cell_to_facet
import numpy as np
import numpy.typing as npt


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


def define_topology(topology: npt.NDArray[np.int64], cell_type: CellType, mesh_element: ET.Element,
                    filename: Path):
    topology_el = ET.SubElement(mesh_element, "Topology")
    topology_el.attrib["NumberOfElements"] = str(topology.shape[0])
    topology_el.attrib["TopologyType"] = str(
        XDMFCellType.from_value(cell_type))
    topology_el.attrib["NodesPerElement"] = str(topology.shape[1])
    it0 = ET.SubElement(topology_el, "DataItem")
    it0.attrib["Dimensions"] = f"{topology.shape[0]} {topology.shape[1]}"
    it0.attrib["Format"] = "HDF"
    it0.text = str(filename.with_suffix(".h5")) + \
        f":/Step0/Connectivity_{str(cell_type)}"


def write_mesh(mesh: Mesh, filename: str | Path):
    filename = Path(filename)

    xdmf = ET.Element("XDMF")
    xdmf.attrib["Version"] = "3.0"
    xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
    domain = ET.SubElement(xdmf, "Domain")

    # Define mesh topology
    grid = ET.SubElement(domain, "Grid")
    grid.attrib["GridType"] = "Uniform"
    grid.attrib["Name"] = "Mesh"
    define_topology(mesh.topology, mesh.cell_type, grid, filename)

    # Define mesh geometry
    geometry = ET.SubElement(grid, "Geometry")
    geometry.attrib["GeometryType"] = "XY" if mesh.geometry.shape[1] == 2 else "XYZ"
    it0 = ET.SubElement(geometry, "DataItem")
    it0.attrib["Dimensions"] = f"{mesh.geometry.shape[0]} {mesh.geometry.shape[1]}"
    it0.attrib["Format"] = "HDF"
    it0.text = str(filename.with_suffix(".h5")) + ":/Step0/Points"

    # Define facet topology and geometry
    facet_grid = ET.SubElement(domain, "Grid")
    facet_grid.attrib["GridType"] = "Uniform"
    facet_grid.attrib["Name"] = "Facet_Mesh"
    define_topology(mesh.facet_topology,
                    cell_to_facet[mesh.cell_type], facet_grid, filename)
    facet_geometry = ET.SubElement(facet_grid, "Geometry")
    facet_geometry.attrib["GeometryType"] = "XY" if mesh.geometry.shape[1] == 2 else "XYZ"
    it0 = ET.SubElement(facet_geometry, "DataItem")
    it0.attrib["Dimensions"] = f"{mesh.geometry.shape[0]} {mesh.geometry.shape[1]}"
    it0.attrib["Format"] = "HDF"
    it0.text = str(filename.with_suffix(".h5")) + ":/Step0/Points"

    # Add facet values
    attrib = ET.SubElement(facet_grid, "Attribute")
    attrib.attrib["Name"] = "Facet markers"
    attrib.attrib["AttributeType"] = "Scalar"
    attrib.attrib["Center"] = "Cell"
    it1 = ET.SubElement(attrib, "DataItem")
    it1.attrib["Dimensions"] = f"{len(mesh.facet_values)}"
    it1.attrib["Format"] = "HDF"
    it1.attrib["DataType"] = "Int"
    it1.text = str(filename.with_suffix(".h5"))+":/Step0/Facet_Markers"
    with open(filename, "w") as outfile:
        outfile.write(
            '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        outfile.write(ET.tostring(xdmf, encoding="unicode"))

    # Create ADIOS2 reader
    assert MPI.COMM_WORLD.size == 1, "Mesh convert only works in serial for now"
    adios = adios2.ADIOS(MPI.COMM_WORLD)
    io = adios.DeclareIO("Mesh writer")
    io.SetEngine("HDF5")
    outfile = io.Open(str(filename.with_suffix(".h5")), adios2.Mode.Write)
    pointvar = io.DefineVariable(
        "Points", mesh.geometry,
        shape=[mesh.geometry.shape[0], mesh.geometry.shape[1]],
        start=[0, 0], count=[mesh.geometry.shape[0], mesh.geometry.shape[1]])
    outfile.Put(pointvar, mesh.geometry)

    topology_var = io.DefineVariable(
        f"Connectivity_{str(mesh.cell_type)}", mesh.topology,
        shape=[mesh.topology.shape[0], mesh.topology.shape[1]],
        start=[0, 0], count=[mesh.topology.shape[0], mesh.topology.shape[1]])
    outfile.Put(topology_var, mesh.topology)

    facet_topology_var = io.DefineVariable(
        f"Connectivity_{str(cell_to_facet[mesh.cell_type])}", mesh.facet_topology,
        shape=[mesh.facet_topology.shape[0], mesh.facet_topology.shape[1]],
        start=[0, 0], count=[mesh.facet_topology.shape[0], mesh.facet_topology.shape[1]])
    outfile.Put(facet_topology_var, mesh.facet_topology)

    facet_values_var = io.DefineVariable(
        f"Facet_Markers", mesh.facet_values,
        shape=[mesh.facet_values.shape[0]],
        start=[0], count=[mesh.facet_values.shape[0]])
    outfile.Put(facet_values_var, mesh.facet_values)

    outfile.PerformPuts()
    outfile.Close()
    assert adios.RemoveIO("Mesh writer")

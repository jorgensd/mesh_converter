# Meshio input/output

A mesh converter from EXODUS 2 to XDMF
Supports reading Facet-markers and Cell-markers from EXODUS 2 into an XDMFFile that can be read by DOLFINx

## Installation
Install the package by calling 
```bash
python3 -m pip install .
```

The package requires ADIOS2 for writing data.

## Example

```python
from mesh_converter import read_exodus2_data, write_mesh

in_mesh = read_exodus2_data("test_mesh_2_vols.e")
write_mesh(in_mesh, "out_mesh2.xdmf")
```
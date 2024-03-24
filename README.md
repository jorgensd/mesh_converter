# A mesh converter from EXODUS 2 to XDMF

Supports reading Facet-markers and Cell-markers from EXODUS 2 into an XDMFFile that can be read by DOLFINx.
Supports both facet blocks and side sets from cubit

## Installation

Install the package by calling

```bash
python3 -m pip install .
```

The package requires ADIOS2 for writing data.

### Conda

If you use conda as an environment, please install all dependencies with

```bash
conda install -c conda-forge adios2=*=mpi_* mpich mpi4py netcdf4
```

and then install this package with

```bash
python3 -m pip install -e . --no-deps
```

## Example

```python
from mesh_converter import read_exodus2_data, write_mesh

in_mesh = read_exodus2_data("test_mesh_2_vols_blocks_and_sidesets.e")
write_mesh(in_mesh, "out_mesh2.xdmf")
```

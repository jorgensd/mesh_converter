from mesh_converter import read_exodus2_data, xdmf, vtk
from pathlib import Path

cwd = Path.cwd()
mesh_files = [file for file in cwd.iterdir() if file.suffix == ".e"]

for file in mesh_files:
    print(file)
    in_mesh = read_exodus2_data(file)
    out_path = (file.parent / "converted" / file.stem).with_suffix(".xdmf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xdmf.write(in_mesh, out_path)
    vtk.write(in_mesh, out_path.with_suffix(".vtkhdf"))
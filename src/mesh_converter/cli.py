import argparse
from mesh_converter import read_exodus2_data, xdmf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an ExodusII mesh file to XDMF format."
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path to the input ExodusII mesh file."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output XDMF mesh file."
    )
    args = parser.parse_args()

    in_mesh = read_exodus2_data(args.input)
    xdmf.write(in_mesh, args.output)
    print(f"Converted {args.input} to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

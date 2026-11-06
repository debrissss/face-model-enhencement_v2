"""单组或批量已配准人脸模型区域对比。"""
import argparse
from pathlib import Path
from .config import InputGroup, load_config, load_manifest
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("mesh", "reference", "rgb", "camera", "manifest"):
        run.add_argument(f"--input_{name}", type=Path)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--task", help="Explicit development verification purpose, e.g. test_regional_surface_comparison")
    args = parser.parse_args(argv)
    singles = [args.input_mesh, args.input_reference, args.input_rgb, args.input_camera]
    if args.input_manifest is not None:
        if any(path is not None for path in singles):
            parser.error("--input_manifest is mutually exclusive with single-group inputs.")
        groups = load_manifest(args.input_manifest)
    else:
        if not all(path is not None for path in singles):
            parser.error("Provide --input_mesh, --input_reference, --input_rgb and --input_camera together.")
        paths = [path.resolve() for path in singles]
        # 沿用数据组/mesh_largeV/model.obj 的项目布局；直接位于组目录的 mesh 也可使用。
        directory = paths[0].parent.parent if paths[0].parent.name.startswith("mesh_") else paths[0].parent
        groups = [InputGroup(directory, *paths)]
    pipeline = Pipeline(load_config(args.config), args.output, args.task)
    pipeline.run(groups)
    if not pipeline.output_complete:
        parser.exit(1, f"Comparison output is incomplete; see {pipeline.directory / 'run.log'}\n")


if __name__ == "__main__":
    main()

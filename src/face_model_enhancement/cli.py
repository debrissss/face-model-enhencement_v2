"""解析独立输入路径，分别读取数据并启动使用明确配置的内存流水线。"""
import argparse
from pathlib import Path

from .config import load_config
from .io.inputs import read_camera, read_depth, read_mesh, read_normal, read_rgb
from .models import PipelineData
from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> None:
    """解析运行命令，校验图像尺寸与相机标定一致后执行流水线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Detect regions, smooth facial skin, constrain frontal normals and brow depth, optionally compare 3dMD (mm).")
    for name in ("mesh", "rgb", "normal", "camera"):
        run.add_argument(f"--input_{name}", required=True, type=Path)
    run.add_argument("--input_depth", type=Path, help="Native relative depth NPZ (raw, valid) for brow_depth_constraint.")
    run.add_argument("--input_3dMD", type=Path, help="Same-subject 3dMD OBJ for the result_comparison stage.")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if any(stage.name == "result_comparison" for stage in config.stages) and args.input_3dMD is None:
        parser.error("result_comparison requires --input_3dMD.")
    if any(stage.name == "brow_depth_constraint" for stage in config.stages) and args.input_depth is None:
        parser.error("brow_depth_constraint requires --input_depth.")
    # 每个命名输入分别经过对应读取器，再组装为阶段共享的数据包。
    data = PipelineData(mesh=read_mesh(args.input_mesh), rgb=read_rgb(args.input_rgb),
                        normal=read_normal(args.input_normal), camera=read_camera(args.input_camera),
                        source_name=f"{args.input_mesh.parent.parent.name}-{args.input_mesh.stem}",
                        gt=read_mesh(args.input_3dMD) if args.input_3dMD is not None else None,
                        depth=read_depth(args.input_depth) if args.input_depth is not None else None)
    size = data.camera.height, data.camera.width
    # 原始相机标定与两种图像必须使用同一像素平面。
    if data.rgb.pixels.shape != (*size, 3) or data.normal.vectors.shape != (*size, 3):
        raise ValueError("RGB, normal map and camera must have matching native image dimensions.")
    if data.depth is not None and data.depth.raw.shape != size:
        raise ValueError("Depth and camera must have matching native image dimensions.")
    inputs = {name: getattr(args, f"input_{name}") for name in ("mesh", "rgb", "normal", "camera")}
    if args.input_3dMD is not None:
        inputs["3dMD"] = args.input_3dMD
    if args.input_depth is not None:
        inputs["depth"] = args.input_depth
    pipeline = Pipeline(config, args.output, input_paths=inputs)
    pipeline.run(data)


if __name__ == "__main__":
    main()


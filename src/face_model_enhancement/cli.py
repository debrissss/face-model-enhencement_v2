"""解析独立输入路径，分别读取数据并启动使用明确配置的内存流水线。"""
import argparse
import os
from pathlib import Path

from .config import load_config
from .io.inputs import read_camera, read_depth, read_mesh, read_normal, read_rgb
from .models import PipelineData


def main(argv: list[str] | None = None) -> None:
    """解析运行命令，校验图像尺寸与相机标定一致后执行流水线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Detect regions, smooth facial skin, constrain frontal normals and brow depth, optionally compare 3dMD (mm).")
    run.add_argument("--input_mesh", required=True, type=Path)
    for name in ("rgb", "normal", "camera"):
        run.add_argument(f"--input_{name}", type=Path)
    run.add_argument("--input_point_clouds", type=Path, help="Raw multi-view PLY and paired camera JSON directory.")
    run.add_argument("--input_multi_view_rgb", type=Path, help="Native per-view RGB directory; absent images use cloud projection.")
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
    names = [stage.name for stage in config.stages]
    if "multi_view_point_cloud_alignment" in names and args.input_point_clouds is None:
        parser.error("multi_view_point_cloud_alignment requires --input_point_clouds.")
    if any(name != "multi_view_point_cloud_alignment" for name in names):
        for name in ("rgb", "normal", "camera"):
            if getattr(args, f"input_{name}") is None:
                parser.error(f"Enhancement stages require --input_{name}.")
    for spec in config.stages:
        if spec.name == "multi_view_point_cloud_alignment":
            os.environ["OMP_NUM_THREADS"] = str(spec.config.threads)
    from .io.point_clouds import read_multi_view_rgb, read_point_clouds
    from .pipeline import Pipeline
    # 每个命名输入分别经过对应读取器，再组装为阶段共享的数据包。
    data = PipelineData(mesh=read_mesh(args.input_mesh, preserve_ply_precision="multi_view_point_cloud_alignment" in names), rgb=read_rgb(args.input_rgb) if args.input_rgb is not None else None,
                        normal=read_normal(args.input_normal) if args.input_normal is not None else None,
                        camera=read_camera(args.input_camera) if args.input_camera is not None else None,
                        source_name=args.input_mesh.parent.parent.name,
                        gt=read_mesh(args.input_3dMD) if args.input_3dMD is not None else None,
                        depth=read_depth(args.input_depth) if args.input_depth is not None else None,
                        point_clouds=read_point_clouds(args.input_point_clouds) if args.input_point_clouds is not None else None,
                        multi_view_rgb=read_multi_view_rgb(args.input_multi_view_rgb) if args.input_multi_view_rgb is not None else None)
    if data.camera is not None:
        size = data.camera.height, data.camera.width
        if data.rgb is not None and data.rgb.pixels.shape != (*size, 3):
            raise ValueError("RGB and camera must have matching native image dimensions.")
        if data.normal is not None and data.normal.vectors.shape != (*size, 3):
            raise ValueError("Normal map and camera must have matching native image dimensions.")
        if data.depth is not None and data.depth.raw.shape != size:
            raise ValueError("Depth and camera must have matching native image dimensions.")
    inputs = {name: getattr(args, f"input_{name}") for name in ("mesh", "rgb", "normal", "camera", "point_clouds", "multi_view_rgb")
              if getattr(args, f"input_{name}") is not None}
    if args.input_3dMD is not None:
        inputs["3dMD"] = args.input_3dMD
    if args.input_depth is not None:
        inputs["depth"] = args.input_depth
    pipeline = Pipeline(config, args.output, input_paths=inputs)
    pipeline.run(data)


if __name__ == "__main__":
    main()


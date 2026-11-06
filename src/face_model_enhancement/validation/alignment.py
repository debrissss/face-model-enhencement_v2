"""逐点验证迁移阶段的导出结果；参考数据只由验收脚本读取。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(points, camera):
    k = camera["Cointrinsics"]
    return points[:, :2] / points[:, 2:3] * [-k["fx"], k["fy"]] + [k["ppx"], k["ppy"]]


def validate(session: Path, stage: Path) -> dict:
    source = session / "multi_view_point_clouds"
    reference = session / "multi_view_point_clouds_aligned"
    actual = stage / "multi_view_point_clouds_aligned"
    report = {"session": session.name, "stage": str(stage.resolve()), "views": {}, "passed": True}
    metrics = json.loads((stage / "alignment_report.json").read_text())["views"]
    for path in sorted(source.glob("*.ply")):
        name = path.stem
        raw, ref, out = [o3d.io.read_point_cloud(str(folder / path.name)) for folder in (source, reference, actual)]
        raw_points, ref_points, points = [np.asarray(cloud.points) for cloud in (raw, ref, out)]
        finite = np.isfinite(raw_points).all(axis=1)
        raw_points = raw_points[finite]
        old, expected, camera = [json.loads((folder / f"{name}.json").read_text()) for folder in (source, reference, actual)]
        transform = np.array(camera["Transforms"]["View_to_Model"])
        inverse = np.array(camera["Transforms"]["Model_to_View"])
        same_count = len(points) == len(ref_points) == len(raw_points)
        point_error = float(np.linalg.norm(points - ref_points, axis=1).max()) if same_count else None
        matrix_error = float(np.max(abs(transform - np.array(expected["Transforms"]["View_to_Model"]))))
        inverse_error = float(np.max(abs(inverse - np.array(expected["Transforms"]["Model_to_View"]))))
        same_rgb = (raw.has_colors() == ref.has_colors() == out.has_colors()
                    and np.array_equal(np.asarray(raw.colors)[finite] if raw.has_colors() else np.asarray(raw.colors), np.asarray(out.colors))
                    and np.array_equal(np.asarray(ref.colors), np.asarray(out.colors)))
        same_normals = raw.has_normals() == out.has_normals() == ref.has_normals()
        normal_error = 0.
        if same_normals and raw.has_normals() and same_count:
            normal_error = float(np.max(abs(np.asarray(out.normals) - np.asarray(ref.normals))))
        calibration_same = all(old[k] == camera[k] == expected[k] for k in ("IRintrinsics", "Cointrinsics", "Extrinsics"))
        coordinate_system_same = camera["CoordinateSystem"] == expected["CoordinateSystem"]
        calibration_transform_same = camera["Transforms"]["IR_to_Color"] == expected["Transforms"]["IR_to_Color"]
        mapped = raw_points @ transform[:3, :3].T + transform[:3, 3]
        transform_error = float(np.linalg.norm(mapped - points, axis=1).max()) if same_count else None
        restored = points @ inverse[:3, :3].T + inverse[:3, 3]
        valid = (abs(raw_points[:, 2]) > 1) & (abs(restored[:, 2]) > 1) if same_count else None
        projection_error = float(np.linalg.norm(project(raw_points[valid], old) - project(restored[valid], old), axis=1).max()) if same_count else None
        inverse_identity_error = float(np.max(abs(transform @ inverse - np.eye(4))))
        passed = bool(same_count and same_rgb and same_normals and normal_error <= 1e-8 and calibration_same and coordinate_system_same and calibration_transform_same
                      and point_error <= 1e-7 and matrix_error <= 1e-8 and inverse_error <= 1e-8
                      and transform_error <= 1e-7 and projection_error <= 1e-6 and inverse_identity_error <= 1e-8
                      and metrics[name]["status"] == "review_candidate")
        report["views"][name] = {"passed": passed, "point_count": len(points), "point_count_matches": same_count,
            "point_max_error_mm": point_error, "matrix_max_error": matrix_error, "inverse_max_error": inverse_error,
            "rgb_preserved": same_rgb, "normal_presence_matches": same_normals, "normal_max_error": normal_error,
            "calibration_preserved": calibration_same, "coordinate_system_matches": coordinate_system_same,
            "ir_to_color_matches": calibration_transform_same, "transform_max_error_mm": transform_error,
            "projection_roundtrip_max_error_px": projection_error, "inverse_identity_max_error": inverse_identity_error,
            "face_detection": metrics[name]["face_detection"]["status"], "color_icp_accepted": metrics[name]["color_refinement"]["accepted"],
            "source_sha256": digest(path), "reference_sha256": digest(reference / path.name), "output_sha256": digest(actual / path.name)}
        report["passed"] &= passed
    report["passed"] &= set(report["views"]) == {p.stem for p in reference.glob("*.ply")} == {p.stem for p in actual.glob("*.ply")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.session, args.stage)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"session": report["session"], "passed": report["passed"], "report": str(args.output)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

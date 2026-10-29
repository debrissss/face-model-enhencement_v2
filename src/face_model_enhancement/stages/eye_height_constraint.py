"""按原始眼睑对应关系，限制法线积分后的眼球高度；长度单位为 mm。"""
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from scipy.spatial import cKDTree

from .base import BaseStage
from ..common.regions import EYE_LOOPS, RIGHT_GLOBE, RIGHT_LID, LEFT_GLOBE, LEFT_LID
from ..config import EyeHeightConfig
from ..models import Camera, ConstraintResult, FloatArray, IntArray, PipelineData


@dataclass(frozen=True)
class EyeReference:
    """保存同眼眼球顶点、眼睑参考顶点及权重，供法线拟合后计算高度上限。

    顶点编号和权重按原始网格确定，后续仅读取更新后的眼睑高度，不重新选点。
    """
    name: str  # right_eye 或 left_eye。
    globe: IntArray  # (G,)，原网格眼球顶点编号。
    lids: IntArray  # (G,K)，先放上轮廓查询结果，再放下轮廓查询结果，每次至多三个点。
    weights: FloatArray  # (G,K)，与 lids 对应；每行权重之和为 1。


def build_eye_references(data: PipelineData) -> tuple[EyeReference, ...]:
    """按上下轮廓对应位置，在同眼全部眼睑顶点中选择参考点并计算权重。

    不对候选点做上下分组；两次查询可使用同一顶点。返回值保留原网格顶点编号。
    """
    # 1. 读取区域结果并投影原始网格顶点；后续邻近查询均使用二维像素距离。
    assert data.regions is not None
    regions = data.regions
    pixels, _ = data.camera.project(data.mesh.vertices)
    references = []

    # 2. 按被摄者右眼、左眼的顺序分别处理，各眼标签与轮廓一一对应。
    eyes = (("right_eye", RIGHT_GLOBE, RIGHT_LID),
            ("left_eye", LEFT_GLOBE, LEFT_LID))
    for (name, globe_label, lid_label), loop in zip(eyes, EYE_LOOPS):
        # 3. 提取当前眼球的原始顶点编号，没有眼球顶点时跳过该眼。
        globe = np.flatnonzero(regions.vertex_labels == globe_label)
        if not len(globe):
            continue

        # 4. 提取同向排列的上下关键点轮廓，并为同眼全部眼睑点建立共用查询索引。
        # 候选点不按上下分组，另一只眼的顶点不参与查询。
        upper = regions.landmarks[list(loop[:9])]
        lower = regions.landmarks[[loop[0], *loop[:7:-1]]]
        lid_ids = np.flatnonzero(regions.vertex_labels == lid_label)
        lid_tree = cKDTree(pixels[lid_ids])

        # 5. 以起始眼角为原点、另一眼角方向为单位眼轴，建立眼部局部像素坐标系。
        # s 沿眼轴，t 垂直眼轴；因此眼睛在图像中倾斜时仍可使用同一规则。
        origin = upper[0]
        axis = upper[-1] - origin
        length = np.linalg.norm(axis)
        axis = axis / length
        across = np.array([-axis[1], axis[0]])
        s = (pixels[globe] - origin) @ axis
        t = (pixels[globe] - origin) @ across

        # 6. 在各眼球点的眼轴位置，分别插值上下轮廓，得到两次查询的目标像素位置。
        targets, ordinates = [], []
        for curve in (upper, lower):
            # 按眼轴坐标排序后插值；仅插值位置限制在两眼角之间，目标重建仍使用原 s。
            u, v = (curve - origin) @ axis, (curve - origin) @ across
            order = np.argsort(u)
            ordinate = np.interp(np.clip(s, 0, length), u[order], v[order])
            targets.append(origin + s[:, None] * axis + ordinate[:, None] * across)
            ordinates.append(ordinate)

        # 7. 按眼球点在上下轮廓之间的位置计算两侧占比，下侧占比为 1-upper_weight。
        # 上轮廓处上侧占比为 1，下轮廓处为 0；间距不足 0.25 像素时使用各半。
        # 最终限制在 [0,1]，避免轮廓外的点产生负权重或超过 1 的权重。
        denominator = ordinates[0] - ordinates[1]
        upper_weight = np.full(len(globe), .5)
        np.divide(t - ordinates[1], denominator, out=upper_weight, where=np.abs(denominator) >= .25)
        upper_weight = np.clip(upper_weight, 0, 1)

        # 8. 对上下目标位置分别查询共用候选池中的至多三个最近点，生成最终权重。
        anchors, weights = [], []
        for target, band_weight in zip(targets, (upper_weight, 1 - upper_weight)):
            # 8.1 最近点索引是候选池内的位置，稍后用 lid_ids 还原原网格编号。
            distance, nearest = lid_tree.query(target, k=min(3, len(lid_ids)))
            distance = distance.reshape(len(globe), -1)
            nearest = nearest.reshape(len(globe), -1)

            # 8.2 先用 1/max(距离,0.25像素)² 加权并在侧内归一化，再乘该侧占比。
            # 两次查询可选到同一点；拼接后每个眼球点的全部参考权重之和为 1。
            weight = 1 / np.maximum(distance, .25) ** 2
            weight = weight / weight.sum(axis=1, keepdims=True) * band_weight[:, None]
            anchors.append(lid_ids[nearest])
            weights.append(weight)

        # 9. 拼接上下查询结果并保存编号与权重；法线拟合后仅读取参考点的新高度。
        references.append(EyeReference(name, globe, np.concatenate(anchors, axis=1),
                                       np.concatenate(weights, axis=1)))
    return tuple(references)


def limit_eye_height(vertices: FloatArray, reference: EyeReference, camera: Camera,
                     gap_mm: float) -> FloatArray:
    """按局部眼睑高度上限，计算保持眼球形状所需的整体模型 -Z 平移。

    只移动该眼全部眼球顶点，不改变眼睑、其他区域或任何 X/Y；不受 strength 缩放。
    调用方须保证相机 R33 为正。
    """
    # 1. 读取 R33，用于将相机 Z 方向的超限量换算为模型 Z 下移距离。
    # 模型 Z 减少 d 时，相机 Z 减少 R33×d；正 R33 的条件由调用方保证。
    direction = camera.model_to_view[2, 2]

    # 2. 将拟合后的顶点换算为相机 Z 方向高度，值越大越靠近观察者。
    # 省略相机平移项：参考权重之和为 1，眼球与加权眼睑高度相减时平移抵消。
    height = vertices @ camera.model_to_view[2, :3]

    # 3. 用当前眼睑参考点高度加权求和并减去 gap_mm，得到各眼球点的高度上限。
    # excess 只保留高于上限的正差值，其余为 0。
    limit = np.sum(height[reference.lids] * reference.weights, axis=1) - gap_mm
    excess = np.maximum(height[reference.globe] - limit, 0.)

    # 4. 取所有眼球点中的最大超限量并除以 R33，得到满足全部上限的统一下移量。
    # 若没有超限则下移量为 0；这里不按 strength 再次缩放。
    correction = float(excess.max() / direction)

    # 5. 复制坐标、统一下移整只眼球，保留形状及相对高度差。
    result = vertices.copy()
    result[reference.globe, 2] -= correction
    return result


def constrain_eye_heights(vertices: FloatArray, references: tuple[EyeReference, ...],
                          camera: Camera, gap_mm: float) -> ConstraintResult:
    """限制双眼高度并记录相对法线积分结果的本阶段 Z 位移。"""
    if references and camera.model_to_view[2, 2] <= 1e-6:
        raise ValueError("Eye height limiting requires a positive camera R33 for model-Z updates.")
    result = vertices.copy()
    for reference in references:
        result = limit_eye_height(result, reference, camera, gap_mm)
    return ConstraintResult(result, result[:, 2] - vertices[:, 2])


class EyeHeightConstraintStage(BaseStage[EyeHeightConfig, ConstraintResult]):
    """从法线积分结果取得坐标，串联眼睑参考和高度限制步骤。"""
    name = "eye_height_constraint"
    output_field = "eye_height"

    def run(self, data: PipelineData, config: EyeHeightConfig) -> PipelineData:
        """保留上游法线结果，将本阶段结果提交到独立字段。"""
        if data.regions is None or data.constraint is None:
            raise ValueError("eye_height_constraint requires region_detection and frontal_normal_constraint results.")
        references = build_eye_references(data)
        result = constrain_eye_heights(data.constraint.vertices, references, data.camera, config.eye_height_gap_mm)
        return replace(data, eye_height=result)

    def compute(self, data: PipelineData, config: EyeHeightConfig) -> ConstraintResult:
        """兼容基类接口，编排和数据提交由 run() 完成。"""
        return cast(ConstraintResult, self.run(data, config).eye_height)

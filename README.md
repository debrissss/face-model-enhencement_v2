# Face model enhancement

从正面 RGB 检测眉毛、眼球、上下眼睑和嘴部区域，先平滑面部轮廓内的皮肤，再以已有正面法线图约束眼口，随后独立限制眼球高度，再通过已有相对深度图迁移眉毛相对皮肤的起伏，最后将同名样本的 3dMD 刚体对齐到原始 largeV，生成灰模及正中剖面对比。坐标单位为 **mm**，保留原始 largeV 顶点排列与三角面索引。

## 安装与运行

```bash
UV_PROJECT_ENVIRONMENT=environment uv sync
environment/bin/python scripts/download_landmarker.py
```

模型下载地址为 MediaPipe 官方版本 1 的 Face Landmarker bundle。权重保存在 `data/models/face_landmarker.task`，来源和 SHA256 保存在同名 JSON。算法本身不联网。

使用 Python 3.10–3.12、NumPy 1.x 和 OpenCV 4.11。关键点模型沿用 MediaPipe 的模型包，但推理使用 OpenCV DNN，并明确指定 OpenCV 后端和 CPU 目标；不导入 MediaPipe Python 运行库，也不创建其 OpenGL/Metal 图形服务。CPU 射线检查使用 VTK 静态单元定位器。

```bash
environment/bin/face-model-enhance run \
  --input_mesh data/input/yy/mesh_largeV/model.ply \
  --input_rgb data/input/yy/multi_view_rgb/up.png \
  --input_normal data/input/yy/multi_view_normal/up.png \
  --input_depth data/input/yy/multi_view_depth/up.npz \
  --input_camera data/input/yy/multi_view_point_clouds_aligned/up.json \
  --input_3dMD data/input/yy/mesh_3dMD/model.obj \
  --output artifacts_test \
  --config configs/pipeline.yaml
```

正式运行将 `--output` 改为 `artifacts`。同样的命令可用于 `dyx`、`lty`、`wh`，替换六个输入路径中的样本名即可。`--input_3dMD` 与原始 largeV 使用同一被摄者的数据；输入数据不会被覆盖。不配置 `result_comparison` 时可以省略 `--input_3dMD`；不配置 `brow_depth_constraint` 时可以省略 `--input_depth`。

相机文件必须使用**对齐后**目录中的 JSON，其中包含 `Cointrinsics` 和 `Transforms.Model_to_View`。`multi_view_point_clouds/` 下原始相机 JSON 不包含该变换，不能替代。原始网格直接使用 mm 坐标，不应用 `normalization.json` 的归一化变换。

## 五个 Stage

`region_detection` 使用 OpenCV CPU 后端运行现有 `face_landmarker.task` 中的人脸检测模型和 478 点关键点模型。相机确定图像直立方向，只检测一次；输出的关键点还原到原始 RGB 像素坐标。眉毛沿用参考项目的 MediaPipe 轮廓，不外扩，眼口标签覆盖重叠的眉毛标签。每只眼睛独立生成掩码：原轮廓填充为可见眼球区域（眼白及虹膜），不腐蚀；按欧氏距离外扩（`eye_padding`，默认 18 像素）后减去眼球区域，得到整圈眼睑。`polygon_mask(points, size, padding=0)` 统一处理轮廓填充及外扩，不合并两眼支持域，也不按眼睛中心分配像素。眼睑是关键点生成的近似区域。嘴部只使用外唇轮廓及其内部，不外扩。区域配置不再使用 `mouth_padding` 和 `brow_padding`，旧配置重跑时需删除这两个参数。区域标签直接投影到原始顶点和面中心，像素坐标向下取整后采样，不执行图外/无效坐标、正深度、面朝向或射线可见性筛选；输入需提供有限且位于图内的投影坐标。原四项检查及阈值保存在 `archive/region_projection_checks.py`，正式区域检测不调用。旧区域配置重跑时需删除 `relative_visibility_tolerance` 和 `min_front_cosine`。其他阶段的法线采样、表面求交和配准逻辑不变。

模型适配位于 `src/face_model_enhancement/common/landmarks.py`。人脸检测使用 128×128 RGB 输入、896 个固定锚点、0.5 置信度和 IoU 阈值，合并最高分人脸的重叠候选；根据眼部关键点旋转并外扩 1.5 倍裁剪框，再以 256×256 RGB 运行关键点模型。检测输入范围为 `[-1,1]`，关键点输入范围为 `[0,1]`。裁剪、边界填充和坐标还原按 MediaPipe v0.10.21 的[图像前处理实现](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/calculators/tensor/image_to_tensor_converter_opencv.cc)及[加权候选合并规则](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/calculators/util/non_max_suppression_calculator.cc)实现。保持模型和关键点编号，不要求不同推理后端产生逐点完全相同的数值。

区域使用一套标签：`0=其他`、`1=右眼球`、`2=右眼睑`、`3=左眼球`、`4=左眼睑`、`5=嘴部`、`6=右眉`、`7=左眉`。左右按被摄者区分；未经镜像的正面图中，右眼通常位于图像左侧。二维像素、网格顶点和面各保存一组对应的 `labels`，不保存 `components` 数组。区域掩码独立生成，统一按眉毛、右眼、左眼、嘴部的顺序赋值，重叠时后写入的覆盖前面的标签。后续阶段在方法内合并同眼的眼球和眼睑标签。旧版本双标签及上下眼睑分开的产物编号不兼容新格式，需从原始输入重新运行；不自动改写旧产物，现有 `model_regions.npz` 不作为 Stage 输入。

`face_smoothing` 位于区域检测与眼口法线约束之间，以 NumPy/SciPy 在 CPU 上计算，不依赖预测法线图。阶段先按表面弯曲和原位置约束求三维整形方向，再沿这些方向求有界位移。处理域以 `FACE_OVAL` 为起点，沿网格边长向外扩展 `domain_padding_mm=6` mm，使侧脸轮廓附近的过渡不再过早压住深点。朝向在 `normal_scale_mm=3` mm 的隐式扩散参考表面上计算，以 `minimum_facing_cosine=-0.4` 保留侧面余量，不直接用每个原始小三角面的朝向筛掉坑壁。眉毛、眼球、眼睑及嘴部标签固定；`protect_nose: true` 时，鼻梁、鼻尖、鼻翼关键点二维凸包内的顶点和接触保护面的顶点也固定。不改写区域检测标签。

平滑内部合并完全相同坐标的顶点用于计算；任一原始副本或关联面属于域外时固定该分析顶点，开放及非流形边界、未引用顶点同样固定。沿网格边长计算至固定顶点的距离，使用五次过渡函数 `u³(10−15u+6u²)`，其中 `u=clip(distance/transition_mm,0,1)`。输出恢复原始顶点排列和面索引，重合副本保持一致；不将合并后的拓扑交给后续 Stage。

正式配置的整形尺度 `fairing_scale_mm` 为 4 mm、最大位移为 2 mm、边界过渡为 2 mm。整形使用原始三角面余切刚度及顶点面积构建线性系统；越靠近固定边界，位置约束越强、位移预算越小。弯曲项只计可移动顶点，固定区域提供边界位置，避免为了消除固定边缘的离散曲率而把网格沿表面拉向边界。`normal_scale_mm` 只影响处理域的朝向估计，不限制最终整形方向。

位移超过局部预算时求解带边界的二次优化，不直接裁剪结果。对实际出现有向投影面积低于原始值 5% 的三角面，局部求解面积约束，同时约束所有与相关顶点相连的面，防止把小面压翻；这不等于全局自交检测。每点相对原始坐标的位移长度不超过 `maximum_displacement_mm × gate`。算法不直接优化颜色，热力图仍以 2 mm 邻域平面拟合残差独立评估；自然曲率和噪声都会计入该指标，不保证全脸为零起伏。

旧引导法线滤波、未使用的法线方向整形及其配置、测试和实验入口已归档到 `archive/guided_face_smoothing_20261029/`。正式阶段不再提供 `method`、`fairing_direction` 或旧引导滤波参数；含这些字段的 YAML 会被拒绝，不执行自动转换。历史运行和配置快照保留原样。当前阶段不执行按样本修复、全局碰撞回退或自动幅度搜索。

`PipelineData.smoothing` 保存完整坐标、XYZ 位移、非负位移长度、固定掩码和过渡权重，全部按原始顶点编号组织。原始 `mesh` 保留不变；眼口阶段以平滑结果为输入并只追加 Z，眼球高度阶段读取法线结果，眉毛阶段继续读取眼球高度结果（省略该阶段时读取法线结果）。各阶段位移表示本阶段增量，眼口位移不包含皮肤平滑。`enhanced_mesh` 优先选择眉毛、眼球高度、法线、平滑中最后已完成的结果。GT 配准及原始剖面始终使用原始输入。

`frontal_normal_constraint` 将相机法线转换到模型坐标系，分别求解两个完整眼部面片和嘴部面片。眼球与眼睑的内部交界不固定。法线积分固定区域边界、与外部面共享的顶点及顶点标签不属于该部件的顶点，仅调整内部顶点的模型 Z；X、Y 不变。

每个三角面使用分片线性高度梯度，对所选区域的全部三角面等权，求解目标 `dz/dx=-nx/nz`、`dz/dy=-ny/nz`。仅使用法线坡度误差和固定边界条件，不计算或应用原始高度先验。迭代时在当前面中心重新采样法线，最后施加增强强度。`prior_weight` 与 `spacing_fraction` 已移除，旧配置重跑时需删除这两个参数。原有先验实现及配置保存在 `archive/frontal_normal_constraint_with_prior_20261027/`。

法线采样直接返回双线性插值、归一化并转换到模型坐标系后的方向，不再按图像边界、相机前后、零法线、有效支持度或法线 Z 分量筛选。`min_normal_z` 及法线观测覆盖率指标已移除，旧配置重跑时需删除该参数。图外采样仍按插值器的常量模式返回零；不对目标坡度的 Z 分母做截断或替换，因此零 Z 可能产生非有限结果。

旧筛选实现、配置与逐条件统计脚本保存在 `archive/normal_observation_filter_20261028/`，正式流程不导入，也不提供开关。四组样本筛选前后的最终增强网格最大坐标差均为 0 mm，来源及区域剖面见 `artifacts_test/feature_normal_filter_profiles_20261028-095124-536334/`；该结论仅覆盖已测样本。

求解中的面积权重和 `use_area_weights` 开关已移除，旧配置重跑时需删除该字段。梯度中的几何分母保持不变。原面积加权实现、配置和消融脚本保存在 `archive/frontal_normal_constraint_with_area_weights_20261027/`；四组历史实验结果保留在 `artifacts_test/normal_area_weight_ablation_20261027-221938-064709/`。

`eye_height_constraint` 独立接在法线约束之后。积分及增强强度混合完成后，按原始眼角连线建立眼轴，在每个可见眼球顶点的同一眼轴位置插值上下眼睑轮廓，分别在同眼全部眼睑顶点中查询距上下轮廓对应位置最近的至多三个原始网格顶点，以像素距离平方倒数加权（距离下限 0.25 像素）。上下眼睑按眼球在眼裂中的位置混合，形成局部高度包络；对应关系在原始网格上确定，包络高度读取积分后的眼睑坐标。沿相机 +Z 定义朝向观察者的高度，要求眼球不超过局部包络减去 `eye_height_gap_mm`（默认 0 mm）。每只眼取全部眼球顶点中的最大正超限量，除以相机旋转的 R33 后，将该眼的全部眼球顶点沿模型 -Z 统一平移。这是保持眼球形状时满足全部高度上限的最小向内平移量，避免逐点截断将曲面压平；它保留法线积分已得到的弧度，不额外拟合球面。若该眼没有超限点则不移动。眼睑、其他区域及全部 XY 坐标在此步骤不变。参数 `eye_height_gap_mm` 移至 `configs/stages/eye_height_constraint.yaml`；旧法线 YAML 中需删除该字段。结果写入 `PipelineData.eye_height`，复用 `ConstraintResult`，位移仅表示本阶段增量；`PipelineData.constraint` 保留纯法线积分结果。此步骤独立于积分增强强度，R33 必须为正；积分时固定的眼球顶点也随同眼曲面整体平移，不扩展到眼睑或其他相邻顶点，不改写区域标签。

法线 PNG 按 `RGB=(normal_camera_XYZ+1)*127.5` 解码并归一化，全白作为无效背景。读取的是 PNG 量化数值，并非原始浮点预测。采样剔除图外、相机后方、无效背景，以及转换后模型 Z 分量不大于配置阈值的观测。

法线阶段不再计算或输出 `reports`，`ConstraintResult` 仅保留顶点坐标和带符号 Z 位移。已移除法线一致性评价、仅用于统计的 XY 投影面积、眼球移动统计和本阶段的 `metrics.json`；梯度算子只返回 `gx、gy`。带报告的实现及配套源码、测试和脚本保存在 `archive/frontal_normal_constraint_with_reports_20261028/`。旧法线阶段 NPZ 含有已移除的 `components` 字段，当前加载器不兼容该格式；需从原始输入重新运行，或使用归档源码读取。眉毛阶段的独立指标继续保留。

法线阶段不处理重合顶点，不做平滑、法线预测、碰撞检测、位移预算、回退或接缝曲率修正。算法遇到错误直接终止并记录上下文，不自动替换算法。

`brow_depth_constraint` 位于法线约束与结果对比之间，输入为法线约束后的网格和 `multi_view_depth/up.npz`。读取 `raw` 浮点相对深度及布尔 `valid`，保持原生 RGB 像素平面，不使用 PNG 预览图或 JSON 中的历史相机路径，不重新预测深度。相机仍由 `--input_camera` 明确提供。

每隔两个像素向网格查询首交点，排除眉眼嘴特征区域和距离可见轮廓不超过 8 像素的位置。按 12 像素网格确定性划分约五分之一的空间留出观测，其余至少 500 个皮肤观测用 15 轮 Huber 加权最小二乘拟合 `depth_mm = slope * raw + shift`。相对深度的符号由拟合系数确定。标定只使用当前源网格，不使用 GT；记录的 `source_holdout_MAE_mm` 是源网格拟合误差，不是 GT 精度。

双眉分别裁剪掩码外扩 6 像素的区域，对网格相机深度和标定预测深度做最小平方五点拉普拉斯延续。眉毛掩码和缺失像素为未知，其他皮肤观测固定；裁剪最外圈的缺失值从最近有效像素填充并记录。两种深度的裁剪有效覆盖率均须达到 85%。将两份延续后的皮肤参考面分别记为 `skin_source` 和 `skin_prediction`，计算：

```text
delta_depth = ((skin_source - source_depth) - (skin_prediction - predicted_depth))
              * strength
```

仅更新当前顶点的模型 Z，X、Y 保持不变。按相机变换换算 `delta_z = -delta_depth / R33`，其中 `R33 = camera.model_to_view[2,2]`；`abs(R33) <= 1e-6` 时无法稳定地用模型 Z 实现深度增量，直接报错。深度目标仍在输入网格的原始投影位置采样一次，不迭代重采样；位移后像素投影可以变化。非眉毛顶点、与非眉毛面共享的顶点、开放及非流形边界固定；可移动顶点直接按 strength 控制起伏迁移，默认强度 0.6，不再施加边界距离权重。超过本 Stage 的 3 mm 欧氏位移预算时统一缩放双眉增量。保留原顶点排列和面索引，不合并重合顶点，不引入朝向检查、自相交检查或自动回退。输入缺失、标定观测不足、眉毛缺失、可移动内部顶点不足或皮肤参考面覆盖不足时直接报错。

边界权重及 `transition_mm` 配置已移除，旧配置重跑时需删除该字段。原实现、配置、测试和消融脚本保存在 `archive/brow_boundary_weights_20261028/`；四组完整流程的有权重/无权重对照保存在 `artifacts_test/brow_boundary_weights_20261028-213255-315405/`。

`PipelineData.constraint` 保留法线结果；`PipelineData.brow_depth` 独立保存眉毛最终顶点、本阶段 XYZ 位移（X、Y 为零，保留原存储格式）和带符号相机深度增量（正值远离相机）、双眉指标及标定信息。两者不原地修改上游网格或深度。

`result_comparison` 在眉毛约束之后计算 GT 对齐和公共剖面。配准只使用原始 largeV 脸部轮廓内的正面未增强区域，排除眉眼口、耳部及颈部；默认按面积确定性采样 20,000 点，以 VTK 最近表面查询执行最多 60 轮点到平面 ICP，每轮保留法线兼容且距离较近的 85% 对应点。先计算 largeV 到 GT 的刚体变换，再取逆，仅变换 GT 副本；不对 GT 拟合尺度，不移动 largeV 或 enhanced，也不使用增强后的结果拟合配准。计算剖面和渲染增强模型时使用眉毛最终结果；兼容旧三阶段配置时使用法线结果。

三视图使用纯白背景、中性灰哑光材质、侧上方主光和反侧补光，仅计算环境光与漫反射，不计算镜面高光。三组模型共用联合取景范围、正交比例、材质和灯光，原始顶点颜色及纹理不进入渲染。行顺序为 `largeV`、`largeV-enhanced`、`3dMD`，列顺序为正、左、右。默认每格 512×512 像素，使用两倍超采样后缩小，合成图为 1536×1632。渲染使用 CPU 射线首交点与插值法线着色，不创建 OpenGL 窗口，不平滑或重建网格几何。

正中纵剖面以原始 RGB 的鼻根关键点 168 在原始 largeV 上的射线交点定位，使用正面相机的横向轴作为平面法向、竖直轴作为高度方向。三组模型使用同一个平面和原点，提取三角面与平面的全部真实交线，不做可见性筛选，不跨断口连接或平滑。三列依次比较 largeV 与 enhanced、largeV 与 3dMD、enhanced 与 3dMD，统一范围并保持横纵等毫米比例；原始为橙色、增强为紫红色、GT 为蓝色，线宽均为 0.3 pt。第一行标注 Full face，横纵坐标分别为相对鼻根的正面轴向深度和高度。第二、三行分别标注 Left eye、Right eye（左右按被摄者区分），使用穿过各眼中心、与全脸剖面平行的纵剖面。眼中心由原始眼角像素中点投射到 largeV 表面确定；各眼剖面的三组模型共享切面和原点，坐标相对该原点。眼部显示半高取双眼原始空间眼宽较大值的 0.65 倍，两眼使用相同放大比例；每行均为三列比较，绘图区宽度与全脸行一致，保持横纵等毫米比例，不筛选交线。`profile_height` 表示每行高度，默认合成图为 2400×3600。

配准指标保存在对比结果的 NPZ 元数据中，其中截尾距离仅描述所选配准样本的拟合情况，不是全脸误差；眼球等局部结构不具有相同的 GT 支持范围。配准失败直接作为阶段计算失败记录，不自动缩放或改用备用算法。

## 配置与输出

`configs/pipeline.yaml` 声明 Stage 顺序和 `pipeline.visualize`；各 Stage 参数位于单独 YAML。所有配置引用的相对路径，以所在 YAML 的目录为基准解析。默认顺序为 `region_detection → face_smoothing → frontal_normal_constraint → eye_height_constraint → brow_depth_constraint → result_comparison`。可省略平滑、眼球高度或眉毛阶段；省略眼球高度阶段时不执行高度限制，也可运行合法序列的非空前缀（包括运行至平滑阶段）；原三阶段和四阶段配置仍可加载。默认启用六个阶段及其可视化。

每次运行自动创建唯一目录，包含 `run_log` 和 `config_snapshot/`。快照同时保留原始配置，以及使用已解析模型路径、可重新加载的配置。日志分别记录算法、序列化和可视化耗时。

仅 `pipeline.visualize` 中的 Stage 保存数组并执行可视化。清空列表时仍返回内存计算结果，磁盘仅保留运行日志和配置快照。可视化失败会记录异常，后续计算继续。

- `region_detection/`：类型化 NPZ、`manifest.json`、`regions.png`（关键点和二维区域）、`regions.ply`（网格面着色）、`regions.json`（标签计数与颜色）。
- `frontal_normal_constraint/`：类型化 NPZ、`manifest.json`、`mesh.ply`（增强网格）、`displacement.png`（原始顶点投影上的带符号 Z 位移）。
- `eye_height_constraint/`：类型化 NPZ、`manifest.json`、`mesh.ply`、`displacement.png`（本阶段带符号 Z 位移），`eye_height.npz` 与法线阶段的 `constraint.npz` 分别保存。
- `face_smoothing/`：类型化 NPZ、`manifest.json`、`mesh.ply`、`displacement.png`（本阶段 XYZ 位移长度，白色为零、红色为最大值）。`smoothing.npz` 中的固定掩码和过渡权重对应原始顶点。
- `brow_depth_constraint/`：类型化 NPZ、`manifest.json`、`mesh.ply`（法线与眉毛增强后的最终网格）、`displacement.png`（本 Stage 的带符号相机深度增量，蓝色向相机、红色远离相机）、`metrics.json`（双眉覆盖率、参考面求解残差、位移及标定指标）。
- `result_comparison/`：类型化 NPZ、`manifest.json`、`render-comparison.png`（三行三列灰模）、`midline-comparison.png`（全脸、左眼、右眼三行，每行三列两两比较）。辅助数值包括 GT 到 largeV 的 4×4 刚体矩阵、对齐后 GT 坐标、公共取景、剖面原点、真实线段和配准残差；该阶段仅生成两张 PNG，不另导出 GT 网格或九张单视图。

NPZ 使用原始顶点及面索引，读取时不需要 pickle。Stage 始终通过 `PipelineData` 交接，可视化使用单独序列化并加载的数据，不接管计算数据。

## 验证

```bash
PYTHONPATH=src environment/bin/python -m unittest discover -s tests -v
```

测试覆盖相机投影、法线旋转与插值、关键点旋转还原、眼口区域、固定边界高度恢复、法线已匹配时的高度保持、区域外保护、类型化读写、配置路径和可视化失败隔离，以及 CPU 检测的锚点顺序、候选合并、裁剪逆变换和纯色图像的无人脸报错。真实 RGB 回归测试使用本项目的模型文件和 `yy` 样本，并禁止导入 MediaPipe 运行库。

自动化测试同时覆盖已知刚体位姿的逆变换恢复、尺度保持、剖面遮挡及断口、对比数据的序列化、禁用 OpenGL 模块后的渲染，以及 `--input_3dMD` 缺失时的 CLI 校验。真实样本运行与测试记录保存在 `artifacts_test/validation.json` 和 `artifacts_test/tests.log`。眉毛回归覆盖稳健标定、皮肤参考面恢复与缺失像素、欧氏预算、旋转相机下的仅 Z 深度换算及 XY 精确保持、区域外保护、上游只读、新类型读写、CLI 输入要求、旧配置兼容及最终结果选择。真实样本验证可重复运行：

```bash
PYTHONPATH=src environment/bin/python scripts/validate_brow_depth.py
```

脚本对四组样本运行至眉毛阶段，`yy` 额外执行完整结果对比。输出与验收记录位于 `artifacts_test/brow_depth_validation/`，其中 `validation.json` 包含各次运行路径、几何保护检查、标定误差及双眉位移指标。该验证不评价眉毛相对 GT 的精度改善。

此前基于已有面标签的独立求解结果仍保留在 `artifacts_test/reference_region_validation/`，用于区分不同验证方式。

此前的 `Could not create an NSOpenGLPixelFormat` 来自 MediaPipe 任务图的图像转张量节点，即使指定 CPU 推理仍请求图形服务。当前通过固定的 OpenCV CPU 推理路径解决，不使用错误后的备用检测策略。旧失败运行日志保留为历史记录。

官方模型下载因域名解析失败未完成，本项目模型文件复用自本机已有 `face_landmarker.task`；同名 JSON 如实记录本地来源和摘要。联网后可以运行下载脚本替换为官方版本。

参考原理来自 `/Users/lym/Documents/Projects/face-model-enhencement` 的眼部与眉毛关键点定义、单视图高度积分、稳健深度标定及双调和皮肤参考起伏迁移；运行时不导入该项目。

阶段共用的算法辅助代码位于 `src/face_model_enhancement/common/`：`geometry` 提供网格几何与面片提取，`regions` 提供区域标签、关键点轮廓与多边形掩码，`landmarks` 提供关键点模型适配，`registration` 提供刚体配准。各调用方直接导入对应子模块。

面部平滑验证可运行：

```bash
MPLCONFIGDIR=/private/tmp/fme-matplotlib environment/bin/python scripts/validate_face_smoothing.py
```

脚本在四组样本上执行完整五阶段流程，输出至 `artifacts_test/face_smoothing_validation/`。`validation.json` 记录运行路径、实际耗时、位移及相邻面法线夹角；各 Run 另存 `smoothing-stages.png`，按原始、平滑、最终三行和正、左、右三列比较。检查覆盖上游只读、固定区域保护、位移预算、重合副本一致性、后续阶段保留皮肤位移、配置快照及类型化读写。法线夹角只描述本次网格的局部变化，不代表相对 GT 的精度改善。

`environment/bin/python scripts/validate_surface_fairing.py` 从上述已有四组产物恢复原始输入，重新执行区域检测和当前正式平滑，不执行后续五官增强。输出到唯一的 `artifacts_test/surface_fairing_<时间>-<标识>/`。它沿用 `artifacts_test/surface_relief/` 的原始 2 mm 检测尺度、原始与旧引导结果及色标，对比新方法的可移动域和内部域蓝色占比，并保存两个三视角灰模/热力图总览、处理后的 PLY、逐顶点数值和验证记录。

使用 `--input_previous <上一版的metrics.json>` 可直接比较上一版整形与当前整形。脚本在原始几何上固定侧脸 ROI，另冻结上一版侧脸指标大于 0.1 mm 的顶点集合，报告当前仍超过该阈值的数量。与新旧各自的可移动域分开统计，避免扩展域之后改变比较对象。

平滑强度对照实验可运行：

```bash
MPLCONFIGDIR=/private/tmp/fme-matplotlib environment/bin/python scripts/compare_fairing_levels.py
```

脚本在四组样本上比较 L1–L4 四档 `fairing_scale_mm=1/2/3/4`，共 16 次完整流程。除平滑尺度外，各档算法配置相同。每次创建唯一的 `artifacts_test/fairing_levels_<时间>-<标识>/`，保存独立 YAML、运行快照、各阶段网格、最终三视角、平滑阶段灰模和同色标热力图，以及汇总数值。旧 A–E 引导滤波实验只保留在归档中。

# Jason-3 SGDR-T NetCDF 数据读取与可视化工具集

针对 `H:\毕业设计\Jason3_Data_Download` 中 Jason-3 卫星雷达高度计
SGDR-T（GDR - Expertise dataset，CNES/EUMETSAT，Processing Baseline G）
NetCDF 数据的通用读取、波形可视化、轨迹地图与中国近海检索工具。

数据规模：约 8700 个 pass 文件，覆盖 cycle 500–513 与 600–624
（2025-01 ~ 2026-02），每个文件约 1 小时弧段、约 6.4 万个 20 Hz 采样点。

---

## 1. 文件清单

| 文件 | 功能 |
|---|---|
| `jason3_reader.py` | **核心读取库**。文件发现、变量解码（_FillValue/scale_factor/add_offset）、时间转换、PassData 容器、自检 |
| `inspect_nc.py` | NetCDF 结构探查（ncdump -h 的 Python 等价物），用于与 Panoply 交叉验证 |
| `plot_waveform.py` | 波形可视化：单点波形 / 多点叠加 / 沿轨瀑布热图 / 波形堆叠图 |
| `plot_track.py` | 轨迹地图：Esri 卫星瓦片底图 + 轨迹线 + 20Hz 采样点（按地表分类或时间着色） |
| `scan_nearshore.py` | 全库扫描，检索中国近海采样区段，输出 `nearshore_scan.csv` |
| `present_nearshore.py` | 一键生成中国近海成品图（总览/放大轨迹 + 瀑布 + 堆叠 + 典型波形对比） |
| `nearshore_scan.csv` | 全库扫描结果明细（8699 个文件的中国海区命中统计） |
| `output/` | 成品图输出目录 |

## 2. 环境要求与安装

Python ≥ 3.10，依赖：

```
pip install h5py numpy matplotlib pillow requests
```

## 3. 重要说明：为什么用 h5py 而不是 netCDF4

**netCDF4-python 在 Windows 上无法打开含中文（非 ASCII）字符的路径。**
`H:\毕业设计\Jason3_Data_Download\...` 这样的目录直接调用
`netCDF4.Dataset(path)` 会报 `FileNotFoundError`（文件实际存在，
是底层 C 库的 Unicode 路径缺陷）。因此本工具集统一使用 **h5py**
（NetCDF4 文件本质是 HDF5 文件，h5py 对 Windows Unicode 路径支持完好），
并自行完成 CF 约定解码。运行任何脚本时若报编码相关错误，
请确保设置了环境变量 `PYTHONUTF8=1`。

## 4. 数据结构（实测，已经 Panoply 口径核对）

```
/ (root)  属性含 title="GDR - Expertise dataset", Conventions=CF-1.7
├── data_01/                    1 Hz 数据
│   ├── time, latitude, longitude, altitude ...
│   ├── ku/   range_ocean, swh_ocean, sig0_ocean, ssha ...
│   └── c/    （C 波段同类变量）
└── data_20/                    20 Hz 数据
    ├── time          float64  units="seconds since 2000-01-01 00:00:00.0" (UTC)
    │                          20Hz 点间隔 ≈0.05094 s（即 20 Hz）
    ├── latitude      int32 微度  scale_factor=1e-6  （北纬为正）
    ├── longitude     int32 微度  scale_factor=1e-6  （0~360°E 约定！）
    ├── surface_classification_flag  int8
    │       0=open_ocean 1=land 2=continental_water 3=aquatic_vegetation
    │       4=continental_ice_snow 5=floating_ice 6=salted_basin
    ├── distance_to_coast        int32 米（负值=位于该产品海岸线库的陆地区）
    ├── angle_of_approach_to_coast int16 scale=0.01 度
    ├── altitude                 int32 off=1.3e6 scale=1e-4 m（椭球高）
    └── ku/
        ├── power_waveform   int32 (N,104) scale_factor=0.001
        │       —— Ku 波段 104 距离门回波功率(count)，本工具集的核心变量
        ├── range_ocean / range_ocean_mle3 / range_adaptive   重定距离
        ├── swh_ocean / sig0_ocean (含 _mle3)                  有效波高/后向散射
        ├── epoch_ocean / amplitude_ocean / noise_floor_ocean  波形拟合参数
        └── peakiness / mqe_ocean / wvf_main_class             波形分类参数
```

与 Panoply 交叉验证：用 Panoply 打开任一 .nc 文件，在
`data_20/ku/power_waveform` 中查看第 1 行数值（Panoply 显示已乘
scale_factor 的解码值），应与 `python jason3_reader.py <文件>` 自检输出
的"波形前 12 门"一致。也可运行
`python inspect_nc.py <文件> --values data_20/ku/power_waveform` 对照。

## 5. 快速上手

```bash
# 0) 读取库自检（打印首点波形数值，可与 Panoply 对照）
python jason3_reader.py

# 1) 看某个文件的结构
python inspect_nc.py "H:/毕业设计/Jason3_Data_Download/cycle_500/JA3_GPS_2PgP500_206_20250130_162932_20250130_172545.nc"

# 2) 单点波形（--mode ocean 自动选开阔海点 / --mode coastal 自动选近海点）
python plot_waveform.py <文件.nc> --mode ocean
python plot_waveform.py <文件.nc> --index 43839

# 3) 沿轨瀑布热图 与 波形堆叠图
python plot_waveform.py <文件.nc> --heatmap --i0 43200 --i1 46000
python plot_waveform.py <文件.nc> --stack   --i0 43200 --i1 46000

# 4) 轨迹地图（Esri 卫星底图；--tiles off 离线；--zoom 指定瓦片级数）
python plot_track.py <文件.nc> --bbox 105 145 0 42 -o track.png

# 5) 全库中国近海扫描（输出 nearshore_scan.csv + 候选排名）
python scan_nearshore.py --workers 10

# 6) 一键生成近海成品图
python present_nearshore.py
```

## 6. 中国近海数据检索结果（2026-09-28 全库扫描）

扫描口径：中国海区框 0–42°N / 100–146°E；"近海点"= 海面分类且离岸 < 50 km。
全库共 8699 个文件，框内 20 Hz 采样点 1806 万个，候选 pass 239 个（近海点 >3000）。

**推荐的一组（本工具集默认展示）**：`cycle 603 / pass 147`
（2025-07-22 03:25–03:32 UTC，夏季无海冰）。上升弧段沿福建—浙江海岸线北上：

| 沿轨索引 | 位置 | 离岸距离 | 波形形态 |
|---|---|---|---|
| 43233 | 23.87°N 台湾海峡陆架 | 61.9 km | 标准 Brown 海洋波形 |
| 43839 | 25.33°N 福建长乐近岸 | **1.9 km** | 近岸畸变：噪声抬高、双峰 |
| 44445 | 26.78°N 浙江沿岸 | 8.8 km | Brown 形态但前沿前移、平台噪声增大 |
| 45051 | 28.22°N 台州陆地 | — | 尖峰波形（幅度约海洋 10 倍） |
| 45657 | 29.98°N 杭州湾口 | 0.4 km | 多峰复杂波形 |

其它优质中国沿岸弧段（同类复现）：浙江 c611p183、c619p219；
渤海湾 c619p230、c605p84；华南（海南—广东）c619p245、c605p99、c613p135；
辽东半岛/北黄海 c500p214（冬季，含海冰类采样点）。
同名 ground track 约每 9.9 天重访一次（如 pass 203 出现于 c504–c512 各 cycle）。

## 7. 常见坑（已在本工具集中处理）

1. **中文路径**：netCDF4 打不开 → 用 h5py（见第 3 节）。
2. **经度 0~360 约定**：直接当 ±180 用会把中国画到太平洋中部；用
   `jason3_reader.lon360_to_180()` 换算。
3. **整型定标**：lat/lon(1e-6)、power_waveform(1e-3)、altitude(1e-4/1.3e6)
   等必须解码后才是物理值，用 `decode_variable()`。
4. **时间基准**：seconds since 2000-01-01 UTC（注意 TAI-UTC=37 s 属性存在，
   `data_20/time` 本身已是 UTC）。
5. **cycle 范围**：数据含 500–513 与 600–624 两段，硬编码 500~513 会漏一半。
6. **地表分类与离岸距离不一致**：两者来自不同的海岸线数据源，近岸
   （尤其岛屿、滩涂、海冰）处可能出现"海面点离岸为负"的现象，属正常。

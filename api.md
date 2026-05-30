# WiFi Fall Guard Backend API

本文档记录当前后端已实现的 REST API 和 WebSocket 接口。

当前默认模式：

- 数据源：ENetFall `.mat` 数据集回放
- 检测器：ENetFall EfficientNet-B0 模型
- CSV 数据源：保留为 fallback/debug 模式
- WebSocket 外层结构保持兼容：

```json
{
  "frame": {},
  "result": {},
  "summary": {},
  "alert_saved": false
}
```

## Base URL

本地开发默认：

```text
http://127.0.0.1:8000
```

WebSocket：

```text
ws://127.0.0.1:8000/ws/csi
```

## 1. 基础接口

### GET /

健康检查。

响应示例：

```json
{
  "app": "WiFi Fall Guard",
  "env": "dev",
  "status": "running"
}
```

### GET /api/status

获取系统状态，包括当前数据源和运行时摘要。

响应示例：

```json
{
  "app": "WiFi Fall Guard",
  "env": "dev",
  "source": {
    "source_mode": "enetfall",
    "current_source": {
      "type": "enetfall_mat",
      "data_dir": "D:\\2026_spring\\IoT\\wifiFall\\data\\ENetFall_dataset_trained_networks",
      "dataset_names": [
        "dataset_home_lab(L).mat",
        "dataset_home_lab(R).mat",
        "dataset_lecture_room.mat",
        "dataset_living_room.mat",
        "dataset_meeting_room.mat"
      ],
      "total_samples": 1234,
      "current_index": 0,
      "device_id": "enetfall-node-001",
      "room": "home",
      "loop": true,
      "window_shape": [3, 625, 30]
    },
    "load_error": null
  },
  "runtime": {
    "total_frames": 0,
    "alert_count": 0,
    "latest_label": null,
    "latest_risk_level": null,
    "uptime_seconds": 1.23
  }
}
```

## 2. 数据源接口

### GET /api/data-source/status

获取当前数据源状态。

响应示例：

```json
{
  "source_mode": "enetfall",
  "current_source": {
    "type": "enetfall_mat",
    "total_samples": 1234,
    "current_index": 0,
    "loop": true,
    "window_shape": [3, 625, 30]
  },
  "load_error": null
}
```

### POST /api/data-source/enetfall

切换到 ENetFall `.mat` 数据源。

请求体：

```json
{
  "data_dir": "D:\\2026_spring\\IoT\\wifiFall\\data\\ENetFall_dataset_trained_networks",
  "dataset_names": [
    "dataset_home_lab(L).mat",
    "dataset_home_lab(R).mat",
    "dataset_lecture_room.mat",
    "dataset_living_room.mat",
    "dataset_meeting_room.mat"
  ],
  "device_id": "enetfall-node-001",
  "room": "home"
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| data_dir | string/null | 否 | ENetFall 数据目录。为空时使用配置 `ENETFALL_DATA_DIR` |
| dataset_names | string[]/null | 否 | 要加载的 `.mat` 文件列表。为空时使用默认 5 个数据集 |
| device_id | string | 否 | 设备 ID，默认 `enetfall-node-001` |
| room | string | 否 | 默认房间名，默认 `home` |

响应示例：

```json
{
  "message": "Data source switched to ENetFall MAT replay",
  "source": {
    "source_mode": "enetfall",
    "current_source": {
      "type": "enetfall_mat",
      "total_samples": 1234,
      "current_index": 0,
      "loop": true,
      "window_shape": [3, 625, 30]
    },
    "load_error": null
  }
}
```

错误：

- `400`：数据目录不存在、`.mat` 文件缺失、字段格式不符合 ENetFall 要求。

### POST /api/data-source/csv

切换到 CSV fallback/debug 数据源。

请求体：

```json
{
  "csv_path": "data/wifi_csi_har_dataset/room_1/1/data.csv",
  "room": "room_1",
  "device_id": "csv-node-001",
  "label": "unknown"
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| csv_path | string | 是 | CSV 文件路径 |
| room | string | 否 | 房间名，默认 `real_room` |
| device_id | string | 否 | 设备 ID，默认 `csv-node-001` |
| label | string | 否 | 无标签列时使用的默认标签 |

`label` 可选值：

```text
empty, walking, sitting, lying, fall, non_fall, unknown
```

响应示例：

```json
{
  "message": "Data source switched to csv",
  "source": {
    "source_mode": "csv",
    "current_source": {
      "type": "csv",
      "csv_path": "data/wifi_csi_har_dataset/room_1/1/data.csv",
      "current_label": "unknown",
      "room": "room_1",
      "device_id": "csv-node-001",
      "subcarrier_count": 64,
      "total_rows": 10000,
      "row_index": 0,
      "loop": true
    },
    "load_error": null
  }
}
```

错误：

- `400`：CSV 文件不存在、CSV 为空、没有可用数值列。

## 3. 模型和检测器接口

### GET /api/model/status

获取 ENetFall 模型状态。

响应示例：

```json
{
  "detector_mode": "enetfall",
  "model_loaded": true,
  "model_name": "efficientnet_b0_enetfall",
  "model_path": "D:\\2026_spring\\IoT\\wifiFall\\data\\ENetFall_dataset_trained_networks\\B0(modified)_trained_with_all_data.pth",
  "device": "cuda",
  "num_classes": 2,
  "class_names": ["non_fall", "fall"],
  "input_shape": [3, 625, 30],
  "load_error": null,
  "active_detector_mode": "enetfall"
}
```

### POST /api/detector/mode

切换检测器模式。

请求体：

```json
{
  "mode": "enetfall"
}
```

可选值：

```text
simple, enetfall
```

说明：

- `enetfall`：使用 EfficientNet-B0 ENetFall 模型。
- `simple`：使用规则检测器，主要用于 CSV/debug fallback。

响应示例：

```json
{
  "message": "Detector mode updated",
  "mode": "enetfall",
  "model": {
    "model_loaded": true,
    "model_name": "efficientnet_b0_enetfall"
  }
}
```

### POST /api/detector/reset

重置检测器状态和运行时缓存。

响应示例：

```json
{
  "message": "Detector and runtime state reset"
}
```

## 4. 检测结果接口

### GET /api/results/latest

获取最新检测结果。

如果 WebSocket 尚未产生任何数据，返回 `404`。

响应示例：

```json
{
  "frame": {
    "frame_id": 1,
    "device_id": "enetfall-node-001",
    "timestamp": 1234567890.0,
    "room": "home_lab_left",
    "subcarriers": [0.001, 0.002, 0.003],
    "simulated_label": "fall",
    "source": "enetfall_mat",
    "window_shape": [3, 625, 30],
    "label": "fall"
  },
  "result": {
    "timestamp": 1234567890.0,
    "room": "home_lab_left",
    "predicted_label": "fall",
    "confidence": 0.96,
    "risk_level": "high",
    "alert": true,
    "reason": "ENetFall EfficientNet-B0 model predicted fall",
    "activity_score": 0.96,
    "features": {
      "model": "efficientnet_b0_enetfall",
      "input_shape": [3, 625, 30],
      "prob_non_fall": 0.04,
      "prob_fall": 0.96,
      "true_label": "fall",
      "source": "enetfall_mat",
      "window_shape": [3, 625, 30]
    }
  }
}
```

### GET /api/results/recent

获取最近检测结果。

查询参数：

| 参数 | 类型 | 默认 | 范围 | 说明 |
|---|---|---:|---|---|
| limit | int | 50 | 1-300 | 返回最近 N 条 |

示例：

```http
GET /api/results/recent?limit=20
```

响应示例：

```json
[
  {
    "frame": {},
    "result": {}
  }
]
```

## 5. 告警接口

### GET /api/alerts

查询告警列表。

查询参数：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---:|---|
| skip | int | 0 | 跳过数量 |
| limit | int | 50 | 返回数量，最大 300 |
| handled | bool/null | null | 是否只看已处理或未处理 |

示例：

```http
GET /api/alerts?skip=0&limit=50&handled=false
```

响应示例：

```json
[
  {
    "event_id": "d4a53b2f-83d0-4c1e-8235-f8a6a7dc2b7f",
    "timestamp": 1234567890.0,
    "room": "home_lab_left",
    "device_id": "enetfall-node-001",
    "predicted_label": "fall",
    "confidence": 0.96,
    "risk_level": "high",
    "activity_score": 0.96,
    "reason": "ENetFall EfficientNet-B0 model predicted fall",
    "handled": false,
    "handler_note": null,
    "created_at": "2026-05-30T12:00:00",
    "updated_at": "2026-05-30T12:00:00"
  }
]
```

### GET /api/alerts/summary/count

获取告警数量统计。

响应示例：

```json
{
  "total": 10,
  "handled": 3,
  "unhandled": 7
}
```

### GET /api/alerts/{event_id}

获取单个告警详情。

响应：

- `200`：返回告警详情。
- `404`：告警不存在。

### PATCH /api/alerts/{event_id}

更新告警处理状态。

请求体：

```json
{
  "handled": true,
  "handler_note": "已确认老人安全"
}
```

响应示例：

```json
{
  "event_id": "d4a53b2f-83d0-4c1e-8235-f8a6a7dc2b7f",
  "handled": true,
  "handler_note": "已确认老人安全",
  "timestamp": 1234567890.0,
  "room": "home_lab_left",
  "device_id": "enetfall-node-001",
  "predicted_label": "fall",
  "confidence": 0.96,
  "risk_level": "high",
  "activity_score": 0.96,
  "reason": "ENetFall EfficientNet-B0 model predicted fall",
  "created_at": "2026-05-30T12:00:00",
  "updated_at": "2026-05-30T12:01:00"
}
```

## 6. WebSocket

### /ws/csi

实时 CSI 检测数据推送。

连接地址：

```text
ws://127.0.0.1:8000/ws/csi
```

推送频率由配置控制：

```env
CSI_FRAME_INTERVAL_MS=100
```

每条消息结构：

```json
{
  "frame": {
    "frame_id": 1,
    "device_id": "enetfall-node-001",
    "timestamp": 1234567890.0,
    "room": "home_lab_left",
    "subcarriers": [0.001, 0.002, 0.003],
    "simulated_label": "fall",
    "source": "enetfall_mat",
    "window_shape": [3, 625, 30],
    "label": "fall"
  },
  "result": {
    "timestamp": 1234567890.0,
    "room": "home_lab_left",
    "predicted_label": "fall",
    "confidence": 0.96,
    "risk_level": "high",
    "alert": true,
    "reason": "ENetFall EfficientNet-B0 model predicted fall",
    "activity_score": 0.96,
    "features": {
      "model": "efficientnet_b0_enetfall",
      "input_shape": [3, 625, 30],
      "prob_non_fall": 0.04,
      "prob_fall": 0.96,
      "true_label": "fall",
      "source": "enetfall_mat",
      "window_shape": [3, 625, 30]
    }
  },
  "summary": {
    "total_frames": 1,
    "alert_count": 1,
    "latest_label": "fall",
    "latest_risk_level": "high",
    "uptime_seconds": 12.3
  },
  "alert_saved": true
}
```

兼容说明：

- `frame.subcarriers` 仍保留，但在 ENetFall 模式下它是轻量 preview，不是完整 `3 x 625 x 30` window。
- 完整 window 不会通过 WebSocket 推送，避免 payload 过大。
- 完整推理输入在后端内部使用。

## 7. 数据模型摘要

### CsiFrame

```json
{
  "frame_id": 1,
  "device_id": "enetfall-node-001",
  "timestamp": 1234567890.0,
  "room": "home_lab_left",
  "subcarriers": [],
  "simulated_label": "fall",
  "source": "enetfall_mat",
  "window_shape": [3, 625, 30],
  "label": "fall"
}
```

### DetectionResult

```json
{
  "timestamp": 1234567890.0,
  "room": "home_lab_left",
  "predicted_label": "fall",
  "confidence": 0.96,
  "risk_level": "high",
  "alert": true,
  "reason": "ENetFall EfficientNet-B0 model predicted fall",
  "activity_score": 0.96,
  "features": {}
}
```

### AlertEventRead

```json
{
  "event_id": "uuid",
  "timestamp": 1234567890.0,
  "room": "home_lab_left",
  "device_id": "enetfall-node-001",
  "predicted_label": "fall",
  "confidence": 0.96,
  "risk_level": "high",
  "activity_score": 0.96,
  "reason": "ENetFall EfficientNet-B0 model predicted fall",
  "handled": false,
  "handler_note": null,
  "created_at": "2026-05-30T12:00:00",
  "updated_at": "2026-05-30T12:00:00"
}
```

## 8. 错误响应

常见错误：

```json
{
  "detail": "error message"
}
```

常见状态码：

| 状态码 | 说明 |
|---:|---|
| 200 | 成功 |
| 400 | 请求参数错误、文件不存在、数据源加载失败 |
| 404 | 查询资源不存在 |
| 422 | 请求体格式不符合 Pydantic Schema |
| 500 | 未处理的服务端错误 |

## 9. 启动与测试

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
uvicorn app.main:app --reload
```

运行测试：

```bash
pytest
```

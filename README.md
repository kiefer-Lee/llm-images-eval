# LLM Images Eval

调用视觉语言大模型（VLM/MLLM）对验证集图片进行目标检测，导出 COCO 风格检测结果，并使用 COCO 指标进行评估。

本项目面向 OpenAI-compatible Chat Completions API 设计，可用于 OpenAI、DashScope/Qwen 兼容模式、vLLM OpenAI Server 以及其它兼容服务。

## 为什么要显式处理图片尺寸

VLM API 往往会在服务端再次缩放图片，而且数据集中每张图片的原始尺寸也可能不同。为了避免坐标系混乱，本项目会显式记录和转换坐标：

1. 读取原图，并记录原始 `width` 和 `height`。
2. 可选地在发送给大模型前 resize 或 letterbox 图片。
3. 在 prompt 中要求模型按指定坐标系返回检测框。
4. 将模型返回的框映射回原始图片坐标系。
5. 按 COCO 格式导出原图像素坐标下的 `[x, y, width, height]`。

坐标模式：

- `--coord-mode sent`：模型返回“实际发送给模型的图片”上的像素框，再由脚本映射回原图像素坐标。这是默认模式。
- `--coord-mode normalized`：模型返回相对于发送图片的 `[0, 1]` 归一化框，再由脚本映射回原图像素坐标。
- `--coord-mode original`：模型直接返回原始数据集图片坐标系下的像素框，脚本只做裁剪和合法性检查，不再缩放。

如果大模型返回的 JSON 无法解析，或 `bbox_2d` 超出了 prompt 约定的坐标范围，脚本会自动重试同一张图片。默认重试 2 次，可用 `--format-retries` 调整。

## 安装

Windows 上 `pycocotools` 用 Conda 安装通常更省心，推荐使用 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate llm-images-eval
```

也可以直接使用 pip：

```powershell
pip install -r requirements.txt
```

## 配置 API

设置 OpenAI-compatible 客户端需要的环境变量：

```powershell
$env:VLM_API_KEY="sk-..."
$env:VLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VLM_MODEL="qwen2.5-vl-72b-instruct"
```

如果使用 OpenAI 官方接口，可以不设置 `VLM_BASE_URL`，只需要把 `VLM_MODEL` 设为支持视觉输入的模型。

## 运行检测

默认只检测 `Datasets/VisDrone` 下的验证集：

```powershell
python -m src.predict `
  --max-images 20 `
  --save-vis
```

如果完整推理时只想随机保存一部分框注图：

```powershell
python -m src.predict `
  --save-vis `
  --vis-random-count 50 `
  --vis-random-seed 42
```

脚本会自动定位：

- `Datasets/VisDrone/annotations/val.json`
- `Datasets/VisDrone/VisDrone2019-DET-val/VisDrone2019-DET-val/images`

完整验证集推理时，去掉 `--max-images` 即可。默认输出到 `outputs/visdrone_val`，脚本会输出：

- `detections.coco.json`：用于评估的 COCO detection 结果列表。推理开始时会创建快照，并在每张图片处理结束后实时原子更新；进程中断时，已完成图片的检测结果仍会保留。
- `raw_responses.jsonl`：每张图的大模型原始回复、prompt 元信息、解析后的检测结果。
- `failures.jsonl`：请求失败或响应解析失败的图片记录。
- `prompt_violation_stats.json`：统计大模型回答不符合 prompt 要求的次数和比例。
- `visualizations/`：加 `--save-vis` 后保存框注可视化结果图；可用 `--vis-random-count` 控制随机保存数量。

推理过程中会显示 tqdm 进度条，并实时展示累计检测框数量、失败图片数量、已保存可视化图片数量。需要减少日志时可加 `--quiet`。

续跑时加 `--resume`。脚本会保留并追加 JSONL 日志，同时从已有 `raw_responses.jsonl` 和 `detections.coco.json` 恢复检测结果，跳过已经成功完成的图片，最后重新写出合并后的 `detections.coco.json`。旧命令里的 `--append-logs` 也会启用同样的恢复逻辑，以兼容已有脚本。

`prompt_violation_stats.json` 包含两个主要比例：

- `violation_attempt_ratio`：不符合 prompt 的回答次数 / 大模型回答总次数。
- `violation_image_ratio`：至少发生过一次不合规回答的图片数 / 处理图片总数。

## 评估

```powershell
python -m src.eval_coco
```

评估脚本底层使用 `pycocotools.COCOeval`，思路上和 MMDetection 常用的 COCO 评估路径一致。

## 转换原始 VisDrone DET 标注

如果手上只有 VisDrone 原始 `.txt` 标注，可以先转换成 COCO JSON：

```powershell
python -m llm_images_eval.tools.visdrone_to_coco `
  --image-dir D:\PythonProjects\SOD\Datasets\VisDrone\VisDrone2019-DET-val\VisDrone2019-DET-val\images `
  --ann-dir D:\PythonProjects\SOD\Datasets\VisDrone\VisDrone2019-DET-val\VisDrone2019-DET-val\annotations `
  --out D:\PythonProjects\SOD\Datasets\VisDrone\annotations\val_from_txt.json
```

如需保留 VisDrone 原始 category 0 ignored regions，可加 `--include-ignored`。转换器会把这些无类别忽略区域展开为每个 VisDrone 有效类别下的 `iscrowd=1` / `ignore=1` 标注，使 `pycocotools.COCOeval` 能按 crowd ignore 逻辑处理它们。

## Prompt 输出约定

默认 prompt 会要求模型只返回 JSON，例如：

```json
{
  "detections": [
    {
      "label": "car",
      "category_id": 4,
      "bbox_2d": [12, 34, 56, 78],
      "score": 0.73,
      "attributes": {}
    }
  ]
}
```

`bbox_2d` 表示 `[x1, y1, x2, y2]`，它所属的坐标系由 `--coord-mode` 决定。导出器会把检测框裁剪到原图范围内，并丢弃非法框或无法匹配到类别的结果。

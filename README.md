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

下面是使用本地 VisDrone COCO 标注文件的示例：

```powershell
python -m llm_images_eval.predict `
  --ann-file D:\PythonProjects\SOD\Datasets\VisDrone\annotations\val.json `
  --image-root D:\PythonProjects\SOD\Datasets\VisDrone\VisDrone2019-DET-val\VisDrone2019-DET-val\images `
  --output-dir D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val `
  --max-images 20
```

完整验证集推理时，去掉 `--max-images` 即可。脚本会输出：

- `detections.coco.json`：用于评估的 COCO detection 结果列表。
- `raw_responses.jsonl`：每张图的大模型原始回复、prompt 元信息、解析后的检测结果。
- `failures.jsonl`：请求失败或响应解析失败的图片记录。

## 评估

```powershell
python -m llm_images_eval.eval_coco `
  --ann-file D:\PythonProjects\SOD\Datasets\VisDrone\annotations\val.json `
  --pred-file D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val\detections.coco.json
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

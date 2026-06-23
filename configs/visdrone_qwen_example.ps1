$env:VLM_API_KEY = "YOUR_API_KEY"
$env:VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VLM_MODEL = "qwen2.5-vl-72b-instruct"

python -m src.predict `
  --output-dir D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val_qwen `
  --image-max-side 1280 `
  --coord-mode sent `
  --json-mode none `
  --save-vis

python -m src.eval_coco `
  --pred-file D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val_qwen\detections.coco.json `
  --output D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val_qwen\metrics.json

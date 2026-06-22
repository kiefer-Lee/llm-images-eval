$env:VLM_API_KEY = "YOUR_API_KEY"
$env:VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VLM_MODEL = "qwen2.5-vl-72b-instruct"

python -m llm_images_eval.predict `
  --ann-file D:\PythonProjects\SOD\Datasets\VisDrone\annotations\val.json `
  --image-root D:\PythonProjects\SOD\Datasets\VisDrone\VisDrone2019-DET-val\VisDrone2019-DET-val\images `
  --output-dir D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val_qwen `
  --image-max-side 1280 `
  --coord-mode sent `
  --json-mode none

python -m llm_images_eval.eval_coco `
  --ann-file D:\PythonProjects\SOD\Datasets\VisDrone\annotations\val.json `
  --pred-file D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val_qwen\detections.coco.json `
  --output D:\PythonProjects\SOD\llm-images-eval\outputs\visdrone_val_qwen\metrics.json

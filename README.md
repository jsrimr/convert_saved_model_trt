# get SAVED_MODEL from TF_HUB and convert it to TensorRT


## Prepare
config.py 수정

## step1 : download SAVED_MODEL from TF_HUB

```python
PYTHONPATH=. python3 classification/download_and_run.py
#or 
PYTHONPATH=. python3 detection_model.py
```

## step2 : convert downloaded model
```
# ex)
python convert_to_trt.py --SAVED_MODEL_DIR=saved_model_detection
```

## step3 : benchmark
```
python3 inference_script.py
```

```bash
#nsys
sudo /opt/nvidia/nsight_systems/nsys profile --trace=cuda,nvtx --delay=60 -o=/sdcard/nsys_results/tensorRT/effnet --force-overwrite=true --export=sqlite python3 inference_script.py 
```

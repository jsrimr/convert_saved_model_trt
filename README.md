# get SAVED_MODEL from TF_HUB and convert it to TensorRT


## Prepare
config.py 수정
## step1 : download SAVED_MODEL from TF_HUB

```python
python3 download_and_run.py
```

## step2 : convert downloaded model
```
python3 convert_to_trt.py
```

## step3 : benchmark
```
python3 inference_script.py
```

```bash
#nsys
sudo /opt/nvidia/nsight_systems/nsys profile --trace=cuda,nvtx --delay=60 -o=/sdcard/nsys_results/tensorRT/effnet --force-overwrite=true --export=sqlite python3 inference_script.py 
```

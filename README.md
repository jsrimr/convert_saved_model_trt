# get SAVED_MODEL from TF_HUB and convert it to TensorRT

## step1 : download SAVED_MODEL from TF_HUB

```python
python download_and_run.py
```

## step2 : convert downloaded model
```
python trt.py
```

## step3 : benchmark
```
python inference_script.py
```
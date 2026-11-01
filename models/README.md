# Models

This directory holds the exported ONNX models. **Nothing here is committed** —
`models/*.onnx` and `models/*.pt` are ignored by Git, because the core
experiment does not need them.

## Do I need anything here?

| Task | Needs a model? |
|---|---|
| Reproducing the search results, tables and figures | **No** — they run from `benchmarks/*.csv` |
| Regenerating latency/memory on a Raspberry Pi | Yes |

## Producing the ONNX files

```bash
pip install ultralytics          # heavy, optional
python models/download_models.py --dry-run
python models/download_models.py --variants yolov5n yolov5s yolov5m
```

That writes:

```text
models/yolov5n_fp32.onnx
models/yolov5s_fp32.onnx
models/yolov5m_fp32.onnx
```

The exports use **dynamic axes**, so one file serves every input resolution in
the search space (320 / 416 / 640).

## Naming convention

`scripts/benchmark_pi.py` looks for `{model}_{precision}.onnx`, e.g.
`yolov5s_fp32.onnx` and `yolov5s_int8.onnx`.

## INT8

INT8 requires a quantised export with a calibration dataset. It is treated as an
**extension**: the core comparison is defined on FP32, and INT8 rows are added
only once they can be produced reproducibly. Do not hand-write INT8 numbers.

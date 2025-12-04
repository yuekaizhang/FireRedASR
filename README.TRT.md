# FireRedASR with TensorRT

This guide explains how to run FireRedASR inference using TensorRT.

## 1. Setup

You need to set up your environment first. You have two options.

**Option 1: Use Docker (Recommended)**

We provide a pre-built Docker image. This is the easiest way.
```bash
docker pull soar97/triton-cosyvoice:25.06
```

**Option 2: Install Manually**

If you do not want to use Docker, you can install the library directly.
```bash
pip install tensorrt_llm==0.20.0
```

## 2. Prepare TRT Engine

Next, download the model and build the TensorRT engine.

First, download the model files from Hugging Face.
```bash
huggingface-cli download yuekai/FireRedASR-AED-L-TensorRT --local-dir FireRedASR-AED-L-TensorRT
```

Then, follow the `FireRedASR-AED-L-TensorRT/export_tensorrt.sh` script to build the engine.
You can find the script [here](https://huggingface.co/yuekai/FireRedASR-AED-L-TensorRT/blob/main/export_tensorrt.sh).

After you finish, your directory should look like this:
```
FireRedASR-AED-L-TensorRT/
├── encoder.fp16.onnx
├── encoder.plan
├── tllm_checkpoint_float16
│   └── decoder
│       ├── config.json
│       └── rank0.safetensors
└── trt_engine_float16
    └── decoder
        ├── config.json
        └── rank0.engine
```

## 3. Run Inference

Now you are ready to run inference.

Use the following script to test the performance.
```bash
cd examples
bash inference_fireredasr_aed_tensorrt.sh
```

## 4. Performance

Here are the results on the AISHELL-1 test set.
This test was run on a single NVIDIA H20 GPU.

-   **PyTorch**: 753 seconds
-   **TensorRT**: 120 seconds

Using TensorRT makes inference much faster.

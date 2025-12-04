#!/usr/bin/env python3

import argparse
import glob
import os
import sys
import time
from fireredasr.models.fireredasr import FireRedAsr


parser = argparse.ArgumentParser()
parser.add_argument('--asr_type', type=str, required=True, choices=["aed", "llm", "aed_tensorrt"])
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument("--tensorrt_model_dir", type=str, default=None, help="directory of tensorrt engines")

# Input / Output
parser.add_argument("--wav_path", type=str)
parser.add_argument("--wav_paths", type=str, nargs="*")
parser.add_argument("--wav_dir", type=str)
parser.add_argument("--wav_scp", type=str)
parser.add_argument("--output", type=str)

# Decode Options
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--decode_max_len", type=int, default=0)
# FireRedASR-AED
parser.add_argument("--nbest", type=int, default=1)
parser.add_argument("--softmax_smoothing", type=float, default=1.0)
parser.add_argument("--aed_length_penalty", type=float, default=0.0)
parser.add_argument("--eos_penalty", type=float, default=1.0)
# FireRedASR-LLM
parser.add_argument("--decode_min_len", type=int, default=0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--llm_length_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)


def main(args):
    wavs = get_wav_info(args)
    fout = open(args.output, "w") if args.output else None

    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir, args.tensorrt_model_dir)

    # Warm-up phase to stabilize inference
    # if len(wavs) > 0:
    #     print("Running warm-up inferences...")
    #     warmup_uttid = [wavs[0][0]]
    #     warmup_wav_path = [wavs[0][1]]
    #     for _ in range(5):  # Number of warm-up runs
    #         _ = model.transcribe(
    #             warmup_uttid,
    #             warmup_wav_path,
    #             {
    #                 "use_gpu": args.use_gpu,
    #                 "beam_size": args.beam_size,
    #                 "nbest": args.nbest,
    #                 "decode_max_len": args.decode_max_len,
    #                 "softmax_smoothing": args.softmax_smoothing,
    #                 "aed_length_penalty": args.aed_length_penalty,
    #                 "eos_penalty": args.eos_penalty,
    #                 "decode_min_len": args.decode_min_len,
    #                 "repetition_penalty": args.repetition_penalty,
    #                 "llm_length_penalty": args.llm_length_penalty,
    #                 "temperature": args.temperature
    #             }
    #         )
    #     print("Warm-up finished.")

    batch_uttid = []
    batch_wav_path = []
    start_time = time.time()
    # wavs = wavs * 100
    for i, wav in enumerate(wavs):
        uttid, wav_path = wav
        batch_uttid.append(uttid)
        batch_wav_path.append(wav_path)
        if len(batch_wav_path) < args.batch_size and i != len(wavs) - 1:
            continue

        results = model.transcribe(
            batch_uttid,
            batch_wav_path,
            {
            "use_gpu": args.use_gpu,
            "beam_size": args.beam_size,
            "nbest": args.nbest,
            "decode_max_len": args.decode_max_len,
            "softmax_smoothing": args.softmax_smoothing,
            "aed_length_penalty": args.aed_length_penalty,
            "eos_penalty": args.eos_penalty,
            "decode_min_len": args.decode_min_len,
            "repetition_penalty": args.repetition_penalty,
            "llm_length_penalty": args.llm_length_penalty,
            "temperature": args.temperature
            }
        )

        for result in results:
            print(result)
            if fout is not None:
                fout.write(f"{result['uttid']}\t{result['text']}\n")

        batch_uttid = []
        batch_wav_path = []

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    with open(args.output + ".time", "w") as f:
        f.write(f"{end_time - start_time} seconds")

def get_wav_info(args):
    """
    Returns:
        wavs: list of (uttid, wav_path)
    """
    base = lambda p: os.path.basename(p).replace(".wav", "")
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
    elif args.wav_scp:
        wavs = [line.strip().split() for line in open(args.wav_scp)]
    elif args.wav_dir:
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        wavs = [(base(p), p) for p in sorted(wavs)]
    else:
        raise ValueError("Please provide valid wav info")
    print(f"#wavs={len(wavs)}")
    return wavs


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)

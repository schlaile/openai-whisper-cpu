#!/usr/bin/python3

import sys
import argparse
import whisper
import torch
import os
from pathlib import Path
from whisper.utils import write_srt
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import json
import time

parser = argparse.ArgumentParser(
    formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("audio",
                    nargs="+", type=str, help="audio file(s) to transcribe")
parser.add_argument("--model",
                    default="medium",
                    choices=whisper.available_models(),
                    help="name of the Whisper model to use")
parser.add_argument("--language", type=str,
                    default=None,
                    choices=sorted(LANGUAGES.keys())
                    + sorted([k.title()
                              for k in TO_LANGUAGE_CODE.keys()]),
                    help="language spoken in the audio, specify None to perform language detection")

parser.add_argument("--output_dir", "-o",
                    type=str, default=".",
                    help="directory to save the outputs")

args = parser.parse_args().__dict__
model_name: str = args.pop("model")
output_dir: str = args.pop("output_dir")
audio_file: str = args.pop("audio")

os.makedirs(output_dir, exist_ok=True)

if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
    if args["language"] is not None:
        warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"


model_fp32 = whisper.load_model(name=model_name, device="cpu")

quantized_model = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

del model_fp32

result = whisper.transcribe(quantized_model, audio_file, **args)

# save SRT
audio_basename = Path(audio_file).stem
with open(os.path.join(output_dir, audio_basename + ".srt"),
          "w", encoding="utf-8") as srt:
    write_srt(result["segments"], file=srt)
        
# save JSON
json_object = json.dumps(result, indent=4)
with open(os.path.join(output_dir, audio_basename + ".json"),
          "w", encoding="utf-8") as output:
    output.write(json_object)


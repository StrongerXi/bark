import os
import numpy as np
import time
import torch
from bark import generate_audio, preload_models, SAMPLE_RATE

# Set things up
preload_models()
text = "In the light of the moon, a little egg lay on a leaf"



# (eager) warm up
original_audio_array = generate_audio(text)

# (eager) benchmark
start = time.time()
original_audio_array = generate_audio(text)
stop = time.time()
generation_duration_s = stop - start
audio_duration_s = original_audio_array.shape[0] / SAMPLE_RATE
print(f"took {generation_duration_s:.0f}s to (eager) generate {audio_duration_s:.0f}s of audio")



# (compile) register torch.compile
generate_audio = torch.compile(generate_audio, backend="eager")

# (compile) warm up, the compilation happens here
start = time.time()
compiled_audio_array = generate_audio(text)
stop = time.time()
generation_duration_s = stop - start
audio_duration_s = compiled_audio_array.shape[0] / SAMPLE_RATE
print(f"took {generation_duration_s:.0f}s to compile _and_ generate {audio_duration_s:.0f}s of audio")

# (compile) benchmark
start = time.time()
compiled_audio_array = generate_audio(text)
stop = time.time()
generation_duration_s = stop - start
audio_duration_s = compiled_audio_array.shape[0] / SAMPLE_RATE
print(f"took {generation_duration_s:.0f}s to (compiled) generate {audio_duration_s:.0f}s of audio")
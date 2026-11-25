"""Unit 4 - P3: Generate an engineering-related image (bridge / robotic system) from a text prompt using a pre-trained text-to-image model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import time
import torch
from diffusers import AutoPipelineForText2Image


def load_pipe():
    """Stable Diffusion Turbo (pre-trained text-to-image model) on CPU - 1 denoising step."""
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", variant="fp16", torch_dtype=torch.float16)
    return pipe.to(torch.float32).to("cpu")


def generate(pipe, prompt, seed=42, size=512):
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, height=size, width=size, generator=g).images[0]

banner("U4 P3 - Text-to-image with Stable Diffusion Turbo")
pipe = load_pipe()
for name, prompt in [("bridge", "a modern cable-stayed bridge over a river at sunset, engineering photograph, highly detailed"),
                     ("robot", "an industrial robotic arm welding a car chassis in a factory, realistic photo")]:
    t = time.time(); img = generate(pipe, prompt)
    path = os.path.join(OUT, f"U4_P03_{name}.png"); img.save(path)
    print(f"Prompt: {prompt}\n  -> saved {os.path.basename(path)} {img.size} in {time.time() - t:.1f}s")

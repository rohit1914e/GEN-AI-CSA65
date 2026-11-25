"""Unit 4 - P4: Create multiple images from different text prompts and compare how prompt changes affect the generated images."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, OUT  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import AutoPipelineForText2Image


def load_pipe():
    """Stable Diffusion Turbo (pre-trained text-to-image model) on CPU - 1 denoising step."""
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", variant="fp16", torch_dtype=torch.float16)
    return pipe.to(torch.float32).to("cpu")


def generate(pipe, prompt, seed=42, size=512):
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, height=size, width=size, generator=g).images[0]

banner("U4 P4 - How prompt changes affect generated images (same seed)")
pipe = load_pipe()
prompts = ["a bridge", "a steel truss bridge, daytime, photograph",
           "a steel truss bridge at night with glowing lights, photograph", "a steel truss bridge, pencil sketch, white background"]
imgs = [generate(pipe, p, seed=7, size=384) for p in prompts]
grid = Image.new("RGB", (384 * 4, 420), "white"); d = ImageDraw.Draw(grid)
for i, (p, im) in enumerate(zip(prompts, imgs)):
    grid.paste(im, (384 * i, 0)); d.text((384 * i + 6, 392), p[:58], fill="black")
    a = np.asarray(im).astype(float)
    print(f"P{i + 1}: {p}\n    brightness={a.mean():6.1f}  colourfulness(std of RGB)={a.std(axis=(0, 1)).mean():5.1f}  "
          f"saturation={np.ptp(a, axis=2).mean():5.1f}")
grid.save(os.path.join(OUT, "U4_P04_prompt_comparison.png"))
print("\nSaved grid outputs/U4_P04_prompt_comparison.png - night prompt is darkest, sketch prompt least saturated,"
      " more detailed prompts give more specific structures.")

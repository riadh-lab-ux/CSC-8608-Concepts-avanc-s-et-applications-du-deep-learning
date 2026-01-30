from __future__ import annotations
from pipeline_utils import to_img2img
import os
from PIL import Image

from pipeline_utils import DEFAULT_MODEL_ID, load_text2img, get_device, make_generator


def save(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def run_text2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42

    # Prompt e-commerce unique (identique pour tous les runs)
    prompt = (
        "ultra-realistic product photo of a backpack on a white background, "
        "studio lighting, soft shadow, very sharp"
    )
    negative = "text, watermark, logo, low quality, blurry, deformed"

    plan = [
        # name, scheduler, steps, guidance
        ("run01_baseline", "EulerA", 30, 7.5),
        ("run02_steps15", "EulerA", 15, 7.5),
        ("run03_steps50", "EulerA", 50, 7.5),
        ("run04_guid4",   "EulerA", 30, 4.0),
        ("run05_guid12",  "EulerA", 30, 12.0),
        ("run06_ddim",    "DDIM",   30, 7.5),
    ]

    for name, scheduler_name, steps, guidance in plan:
        pipe = load_text2img(model_id, scheduler_name)
        device = get_device()
        g = make_generator(seed, device)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )

        img = out.images[0]
        save(img, f"outputs/t2i_{name}.png")
        print("T2I", name, {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance})

def run_img2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    scheduler_name = "EulerA"
    steps = 30
    guidance = 7.5

    # Image source (produit) dans TP2/inputs/
    init_path = "TP2/inputs/jersey.png"

    prompt = "ultra-realistic e-commerce product photo of the same red football jersey on a white background, studio lighting, soft shadow, very sharp"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    strengths = [
        ("run07_strength035", 0.35),
        ("run08_strength060", 0.60),
        ("run09_strength085", 0.85),
    ]

    pipe_t2i = load_text2img(model_id, scheduler_name)
    pipe_i2i = to_img2img(pipe_t2i)

    device = get_device()

    init_image = Image.open(init_path).convert("RGB")
    # (optionnel mais safe) resize en 512x512 pour éviter les soucis de tailles
    init_image = init_image.resize((512, 512))

    for name, strength in strengths:
        # IMPORTANT: reset generator pour garder la même seed sur chaque run
        g = make_generator(seed, device)

        out = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        )
        img = out.images[0]
        save(img, f"TP2/outputs/i2i_{name}.png")
        print("I2I", name, {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance, "strength": strength})

def main() -> None:
    run_img2img_experiments()

if __name__ == "__main__":
    main()


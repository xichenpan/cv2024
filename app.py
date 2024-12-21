import gradio as gr
import numpy as np
import torch

from pipelines.pipeline_soda import SODAPipeline
from models.soda import SODAConfig
from trainer_utils import find_newest_checkpoint
import random
import re
import os

MIN_SEED = 0
MAX_SEED = np.iinfo(np.int32).max
MAX_INPUT_IMAGES = 4
DEFAULT_INPUT_IMAGES = 1
MAX_IMAGES_PER_PROMPT = 4
DEFAULT_IMAGES_PER_PROMPT = 1

# Add preset negative prompts at the top with other constants
PRESET_NEGATIVE_PROMPTS = {
    "None": "",
    "Basic": "low resolution, low quality, blurry",
    "Detailed": "bad anatomy, signature, watermark, username, error, missing limbs, error",
    "Artistic": "photographic, realistic, photo-realistic, sharp focus, 3d render, oversaturated",
}


def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def variable_images(k):
    k = int(k)
    return [gr.update(visible=True)] * k + [gr.update(visible=False)] * (MAX_INPUT_IMAGES - k)


def process_interleaved_vision_language(
    prompt,
    negative_prompt,
    seed,
    guidance_scale,
    text_guidance_scale,
    num_inference_steps,
    num_images_per_prompt,
    *input_images,
):
    # Use the SODA pipeline to generate images
    valid_images = [img for img in input_images if img is not None]

    images = pipeline(
        image=valid_images if len(valid_images) > 0 else None,
        caption=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        text_guidance_scale=text_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.Generator().manual_seed(seed),
        device=device,
        enable_progress_bar=True,
    ).images
    return images


def load_checkpoint(selected_checkpoint):
    checkpoint_path = find_newest_checkpoint(selected_checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}")
    global pipeline
    pipeline = SODAPipeline.from_pretrained(checkpoint_path, ignore_mismatched_sizes=True, _gradient_checkpointing=False)
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    return checkpoint_path.split("/")[-2]


def get_available_checkpoints(base_checkpoint):
    parent_dir = os.path.dirname(base_checkpoint)
    checkpoints = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    return sorted(checkpoints, key=str.lower)


if __name__ == "__main__":
    device = "cuda"
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default="/fsx-project/xichenpan/output/flow_siglip_luminanexticl_1024_trainall_1e5_64_sstkt2ii2t_wocfg_50_32"
    )
    args = parser.parse_args()
    resume_from_checkpoint = args.resume_from_checkpoint
    available_checkpoints = get_available_checkpoints(resume_from_checkpoint)

    with gr.Blocks(fill_width=True) as demo:
        with gr.Row():
            with gr.Column():
                checkpoint_dropdown = gr.Dropdown(
                    choices=available_checkpoints,
                    value=resume_from_checkpoint,
                    label="Select Checkpoint",
                    allow_custom_value=True,  # Enable search functionality
                    filterable=True,  # Make the dropdown searchable
                )
                checkpoint_name = gr.Textbox(label="Checkpoint Name", interactive=True, value=resume_from_checkpoint)
                # Load the selected checkpoint
                load_checkpoint(checkpoint_dropdown.value)

                prompt = gr.Textbox(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Use <i> to represent the images in prompt",
                )
                negative_prompt_preset = gr.Dropdown(
                    choices=list(PRESET_NEGATIVE_PROMPTS.keys()),
                    value="Detailed",
                    label="Negative Prompt Preset",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    max_lines=1,
                    value=PRESET_NEGATIVE_PROMPTS["Detailed"],
                )

                # Add function to update negative prompt textbox when preset is selected
                def update_negative_prompt(preset_name):
                    return PRESET_NEGATIVE_PROMPTS[preset_name]

                negative_prompt_preset.change(fn=update_negative_prompt, inputs=[negative_prompt_preset], outputs=[negative_prompt], queue=False)

                seed = gr.Slider(label="Seed", minimum=MIN_SEED, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                guidance_scale = gr.Slider(1, 30, step=0.5, value=3, label="Guidance Scale")
                text_guidance_scale = gr.Slider(1, 30, step=0.5, value=7.5, label="Text Guidance Scale")
                with gr.Accordion("Advanced options", open=False):
                    num_inference_steps = gr.Slider(1, 100, step=1, value=30, label="Number of Inference Steps")
                    num_images_per_prompt = gr.Slider(
                        1,
                        MAX_IMAGES_PER_PROMPT,
                        value=DEFAULT_IMAGES_PER_PROMPT,
                        step=1,
                        label="Number of Images",
                    )
                generate_btn = gr.Button("Generate Images")
                num_input_images = gr.Slider(
                    1,
                    MAX_INPUT_IMAGES,
                    value=DEFAULT_INPUT_IMAGES,
                    step=1,
                    label="Number of input images:",
                )
                input_images = [
                    gr.Image(
                        label=f"img{i}",
                        type="pil",
                        visible=True if i < DEFAULT_INPUT_IMAGES else False,
                    )
                    for i in range(MAX_INPUT_IMAGES)
                ]
                num_input_images.change(variable_images, num_input_images, input_images)
            with gr.Column():
                output_gallery = gr.Gallery(columns=2, label="Generated Images")

        checkpoint_dropdown.change(fn=load_checkpoint, inputs=checkpoint_dropdown, outputs=checkpoint_name, queue=False)
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process_interleaved_vision_language,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                guidance_scale,
                text_guidance_scale,
                num_inference_steps,
                num_images_per_prompt,
                *input_images,
            ],
            queue=False,
            outputs=output_gallery,
        )

        generate_btn.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process_interleaved_vision_language,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                guidance_scale,
                text_guidance_scale,
                num_inference_steps,
                num_images_per_prompt,
                *input_images,
            ],
            queue=False,
            outputs=output_gallery,
        )

    demo.launch(share=True)

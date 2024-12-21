from typing import Optional, Union, List
from typing import Tuple

import torch
from PIL import Image
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from models.soda import SODA


class SODAPipeline(SODA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]],
        caption: Optional[str] = None,
        negative_prompt: Optional[str] = "",
        guidance_scale: float = 4.0,
        text_guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        output_type: Optional[str] = "pil",
        need_preprocess: bool = True,
        device: Optional[Union[str, torch.device]] = "cuda",
        return_dict: bool = True,
        **kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Handle single inputs
        if image is not None and not isinstance(image, list) and need_preprocess:
            image = [image]
        if caption is not None and not isinstance(caption, list):
            caption = [caption]

        if hasattr(self, "encoder"):
            if image is None:
                x_source = None
            elif need_preprocess:
                x_source = self.encoder.process(image)
            else:
                x_source = image

            if caption is not None:
                try:
                    caption, attn_mask = self.tokenize(caption)
                except:
                    caption, attn_mask = None, None
            else:
                caption, attn_mask = None, None
            image_sizes = None
        elif image is not None:
            caption, attn_mask, x_source, image_sizes = self.tokenize(caption, image)
        else:
            caption, attn_mask = self.tokenize(caption)
            x_source, image_sizes = None, None

        samples = self.sample_images(
            x_source=x_source.to(device=device, dtype=torch.bfloat16) if x_source is not None else None,
            caption=caption.to(device=device) if caption is not None else None,
            negative_prompt=negative_prompt,
            attn_mask=attn_mask.to(device=device, dtype=torch.bfloat16) if attn_mask is not None else None,
            image_sizes=image_sizes.to(device=device) if image_sizes is not None else None,
            guidance_scale=guidance_scale,
            text_guidance_scale=text_guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )
        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)

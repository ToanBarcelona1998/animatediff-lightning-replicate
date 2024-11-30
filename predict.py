import os, sys
from cog import BasePredictor, Input, Path
sys.path.append('/content')
os.chdir('/content')

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from safetensors.torch import load_file

from typing import List, Union
import tempfile
import numpy as np
import PIL.Image
import imageio

def export_to_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 10
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    writer = imageio.get_writer(output_video_path, fps=fps)
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()
    return output_video_path

def export_to_gif(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_gif_path: str = None, fps: int = 10
) -> str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    imageio.mimsave(output_gif_path, video_frames, fps=fps)
    return output_gif_path

def inferenceToVideo(prompt, negative_prompt, guidance_scale, pipe):
    try:
        output = pipe(prompt=prompt, negative_prompt = negative_prompt, guidance_scale=guidance_scale, num_inference_steps=4)
        export_to_video(output.frames[0], "/content/animation.mp4")
    except Exception as error:
        print(f"global exception: {error}")

def inferenceToGif(prompt, negative_prompt, guidance_scale, pipe):
    try:
        output = pipe(prompt=prompt, negative_prompt = negative_prompt, guidance_scale=guidance_scale, num_inference_steps=4)
        export_to_gif(output.frames[0], "/content/animation.gif")
    except Exception as error:
        print(f"global exception: {error}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        device = "cuda"
        dtype = torch.float16
        step = 4
        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file('/content/models/animatediff_lightning_4step_diffusers.safetensors', device=device))
        self.pipe = AnimateDiffPipeline.from_pretrained('/content/models/toonyou_beta6', motion_adapter=adapter, torch_dtype=dtype).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    def predict(
        self,
        prompt: str = Input(default='A girl smiling'),
        negative_prompt: str = Input(default='(worst quality, low quality, letterboxed)'),
        guidance_scale: float = Input(default=1.0),
        type: str = Input(default='gif' , choices = ['video', 'gif'])
    ) -> Path:
        if(type == "video"):
            output_image = inferenceToVideo(prompt, negative_prompt,guidance_scale, self.pipe)
            return Path('/content/animation.mp4')
        else:
            output_image = inferenceToGif(prompt, negative_prompt, guidance_scale, self.pipe)
            return Path('/content/animation.gif')
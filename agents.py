import torch
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from diffusers import CogVideoXPipeline
from TTS.api import TTS
from moviepy import ImageSequenceClip
import soundfile as sf
from openai import OpenAI
from threading import Thread
from utils import *

class Writer:
    def __init__(self, model, device=None):
        self.model = model
        self.client = OpenAI(
            api_key=os.environ["DEEPINFRA_TOKEN"],
            base_url="https://api.deepinfra.com/v1/openai"
        )

    def forward(self, x):
        messages = [
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": x}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
        
class Reviewer:
    def __init__(self, model):
        self.model = model
        self.client = OpenAI()
    def forward(self, x, system_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}
            ],
            temperature=0.0,
        )
        story = response.choices[0].message.content.replace('"', '').replace("“", "").replace("”", "")
        return story

class Narrator:
    def __init__(self, model, device):
        self.device = device
        self.model = TTS(model).to(device)
    def forward(self, x, speaker):
        filename = time.strftime("%Y%m%d-%H%M%S") + "-narration.wav"
        self.model.tts_to_file(text=x,
            file_path=filename,
            speaker_wav=speaker,
            language="en"
        )
        return filename

class MovieDirector:
    def __init__(self, model):
        self.model = model
        self.client = OpenAI()
    def forward(self, x):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIRECTOR_PROMPT},
                {"role": "user", "content": x}
            ],
            temperature=0.0,
        )
        story = response.choices[0].message.content.replace('"', '').replace("“", "").replace("”", "")
        return story

class MusicDirector:
    def __init__(self, model):
        self.model = model
        self.client = OpenAI()
    def forward(self, x):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": COMPOSER_PROMPT},
                {"role": "user", "content": x}
            ],
            temperature=0.0,
        )
        story = response.choices[0].message.content.replace('"', '').replace("“", "").replace("”", "")
        return story
    
class Musician:
    def __init__(self, model, device):
        self.model = MusicgenForConditionalGeneration.from_pretrained(model).to(device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.device = device
    async def forward(self, x):
        inputs = self.processor(
            text=[x],
            padding=True,
            return_tensors="pt"
        )
        audio = self.model.generate(**inputs.to(self.device), max_new_tokens=308)
        audio = audio.cpu().numpy()[0][0]
        audio = audio[:int(6 * self.model.config.audio_encoder.sampling_rate)]
        filename = time.strftime("%Y%m%d-%H%M%S") + "-music.mp3"
        sf.write(filename, audio, self.model.config.audio_encoder.sampling_rate)
        return filename

class Animator:
    def __init__(self, model, device):
        self.pipe = CogVideoXPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16
        ).to(device)
        self.device = device
    async def forward(self, x):
        frames = self.pipe(
            prompt=x,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=48,
            guidance_scale=6,
            generator=torch.Generator(device=self.device).manual_seed(42)
        ).frames[0]
        frames = [np.array(frame) for frame in frames]
        video = ImageSequenceClip(frames, fps=8)
        return video
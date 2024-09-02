import os
import sys
import json
import re
import time
import librosa
import torch
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import threading
import uvicorn
import logging
from multiprocessing import Queue, Process, cpu_count

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define FastAPI app
app = FastAPI()

class GUIConfig:
    def __init__(self) -> None:
        self.pth_path: str = ""
        self.index_path: str = ""
        self.pitch: int = 0
        self.formant: float = 0.0
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time: float = 2.5
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.rms_mix_rate: float = 0.0
        self.index_rate: float = 0.0
        self.n_cpu: int = 4
        self.f0method: str = "fcpe"

class ConfigData(BaseModel):
    pth_path: str
    index_path: str
    threhold: int = -60
    pitch: int = 0
    formant: float = 0.0
    index_rate: float = 0.3
    rms_mix_rate: float = 0.0
    block_time: float = 0.25
    crossfade_length: float = 0.05
    extra_time: float = 2.5
    n_cpu: int = 4
    I_noise_reduce: bool = False
    O_noise_reduce: bool = False
    use_pv: bool = False
    f0method: str = "fcpe"

class Harvest(Process):
    def __init__(self, inp_q, opt_q):
        super(Harvest, self).__init__()
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld
        while True:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)

class AudioAPI:
    def __init__(self) -> None:
        self.gui_config = GUIConfig()
        self.config = None  # Initialize Config object as None
        self.flag_vc = False
        self.function = "vc"
        self.delay_time = 0
        self.rvc = None  # Initialize RVC object as None
        self.inp_q = None
        self.opt_q = None
        self.n_cpu = min(cpu_count(), 8)

    def initialize_queues(self):
        self.inp_q = Queue()
        self.opt_q = Queue()
        for _ in range(self.n_cpu):
            p = Harvest(self.inp_q, self.opt_q)
            p.daemon = True
            p.start()

    def load(self):
        try:
            with open("configs/config.json", "r", encoding='utf-8') as j:
                data = json.load(j)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            data = {
                "pth_path": "",
                "index_path": "",
                "threhold": -60,
                "pitch": 0,
                "formant": 0.0,
                "index_rate": 0,
                "rms_mix_rate": 0,
                "block_time": 0.25,
                "crossfade_length": 0.05,
                "extra_time": 2.5,
                "n_cpu": 4,
                "f0method": "fcpe",
                "use_pv": False,
            }
        return data

    def set_values(self, values):
        logger.info(f"Setting values: {values}")
        if not values.pth_path.strip():
            raise HTTPException(status_code=400, detail="Please select a .pth file")
        if not values.index_path.strip():
            raise HTTPException(status_code=400, detail="Please select an index file")
        self.config.use_jit = False
        self.gui_config.pth_path = values.pth_path
        self.gui_config.index_path = values.index_path
        self.gui_config.threhold = values.threhold
        self.gui_config.pitch = values.pitch
        self.gui_config.formant = values.formant
        self.gui_config.block_time = values.block_time
        self.gui_config.crossfade_time = values.crossfade_length
        self.gui_config.extra_time = values.extra_time
        self.gui_config.I_noise_reduce = values.I_noise_reduce
        self.gui_config.O_noise_reduce = values.O_noise_reduce
        self.gui_config.rms_mix_rate = values.rms_mix_rate
        self.gui_config.index_rate = values.index_rate
        self.gui_config.n_cpu = values.n_cpu
        self.gui_config.use_pv = values.use_pv
        self.gui_config.f0method = values.f0method
        return True

    def start_vc(self):
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.rvc = rvc_for_realtime.RVC(
            self.gui_config.pitch,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            self.rvc if self.rvc else None,
        )
        self.gui_config.samplerate = (
            self.rvc.tgt_sr
            if self.gui_config.sr_type == "sr_model"
            else 16000
        )
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        logger.info("Voice changer loaded!")

    def stop_vc(self):
        self.flag_vc = False
        self.rvc = None
        logger.info("Voice changer stopped.")

# WebSocket endpoint to handle audio streams
@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_api = AudioAPI()
    try:
        audio_api.initialize_queues()
        audio_api.start_vc()
        while True:
            try:
                data = await websocket.receive_bytes()
                # Process the received audio data
                processed_audio = audio_api.process_audio(data)
                # Send the processed audio back to the client
                await websocket.send_bytes(processed_audio)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error during audio processing: {e}")
                await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

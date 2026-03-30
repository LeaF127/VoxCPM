"""
VoxCPM API Service
版本: v2.0.0

基于类和封装思想组织的 API 服务
- ASR 功能仅供内部使用
- TTS 功能对外暴露
"""

import argparse
import io
import json
import os
import shutil
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from funasr import AutoModel
from voxcpm.core import VoxCPM

# ==================== 配置类 ====================

class APIConfig:
    """API 配置类"""
    # ASR 配置
    ASR_MODEL_ID = "iic/SenseVoiceSmall"

    # TTS 配置
    DEFAULT_TTS_MODEL_DIR = "./models/VoxCPM1.5"
    HF_REPO_ID = os.environ.get("HF_REPO_ID", "openbmb/VoxCPM1.5")

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 目录配置
    OUTPUT_DIR = "outputs/tasks"
    PROMPTS_DIR = "prompts"
    UPLOAD_DIR = "uploads/vc_wavs"
    TEMP_DIR = "temp"

    # 推理配置
    INFERENCE_TIMEOUT = 120


# ==================== 模型层 ====================

class BaseModelWrapper(ABC):
    """模型抽象基类"""

    @abstractmethod
    def load(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass


class ASRModel(BaseModelWrapper):
    """ASR 模型封装 - 单例模式

    使用 SenseVoice 模型进行语音识别，供内部 TTS 服务使用
    """
    _instance: Optional["ASRModel"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_id: str = None, device: str = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.model_id = model_id or APIConfig.ASR_MODEL_ID
        self.device = device or APIConfig.DEVICE
        self.model: Optional[AutoModel] = None
        self._initialized = False
        self._lock = threading.Lock()

    def load(self) -> None:
        """加载 ASR 模型（线程安全）"""
        with self._lock:
            if self._initialized:
                return
            print(f"[ASRModel] Loading model: {self.model_id}", file=sys.stderr)
            self.model = AutoModel(
                model=self.model_id,
                disable_update=True,
                log_level='DEBUG',
                device="cuda:0" if self.device == "cuda" else "cpu",
            )
            self._initialized = True
            print("[ASRModel] Model loaded successfully.", file=sys.stderr)

    def recognize(self, audio_path: str) -> str:
        """识别音频文本

        Args:
            audio_path: 音频文件路径

        Returns:
            识别的文本
        """
        if not self._initialized:
            self.load()

        res = self.model.generate(input=audio_path, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text

    def is_loaded(self) -> bool:
        return self._initialized


class TTSModel(BaseModelWrapper):
    """VoxCPM TTS 模型封装 - 单例模式，懒加载

    使用 VoxCPM 模型进行文本转语音
    """
    _instance: Optional["TTSModel"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_dir: str = None, device: str = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.model_dir = model_dir or APIConfig.DEFAULT_TTS_MODEL_DIR
        self.device = device or APIConfig.DEVICE
        self.model: Optional[VoxCPM] = None
        self._initialized = False
        self._lock = threading.Lock()

    def _resolve_model_dir(self) -> str:
        """解析模型目录路径

        优先级：
        1. 本地默认目录
        2. HF repo 下载目录
        3. models 目录
        """
        if os.path.isdir(self.model_dir):
            return self.model_dir

        repo_id = APIConfig.HF_REPO_ID.strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"[TTSModel] Downloading from HF repo '{repo_id}' to '{target_dir}'...", file=sys.stderr)
                    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"[TTSModel] Warning: HF download failed: {e}. Falling back to 'models'.", file=sys.stderr)
                    return "models"
            return target_dir
        return "models"

    def load(self) -> None:
        """懒加载 TTS 模型（线程安全）"""
        with self._lock:
            if self._initialized:
                return
            print("[TTSModel] Model not loaded, initializing...", file=sys.stderr)
            model_dir = self._resolve_model_dir()
            print(f"[TTSModel] Using model dir: {model_dir}", file=sys.stderr)
            self.model = VoxCPM.from_pretrained(hf_model_id=model_dir)
            self._initialized = True
            print(f"[TTSModel] Model loaded successfully. Sample rate: {self.model.tts_model.sample_rate}", file=sys.stderr)

    def generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_len: int = 600,
        normalize: bool = False,
    ) -> np.ndarray:
        """生成语音

        Args:
            text: 目标文本
            prompt_wav_path: 参考音频路径
            prompt_text: 参考文本
            cfg_value: CFG 值
            inference_timesteps: 推理时间步
            max_len: 最大生成长度
            normalize: 是否文本正则化

        Returns:
            音频 numpy 数组
        """
        if not self._initialized:
            self.load()

        wav = self.model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            max_len=int(max_len),
            normalize=normalize,
        )
        return wav

    def get_sample_rate(self) -> int:
        """获取采样率"""
        if not self._initialized:
            self.load()
        return self.model.tts_model.sample_rate

    def is_loaded(self) -> bool:
        return self._initialized


# ==================== 服务层 ====================

class ASRService:
    """ASR 服务 - 内部使用

    封装 ASR 模型，提供音频识别功能
    """

    def __init__(self, model: ASRModel = None):
        self.model = model or ASRModel()

    def transcribe(self, audio_path: str) -> str:
        """转录音频为文本

        Args:
            audio_path: 音频文件路径

        Returns:
            识别的文本，如果音频路径为空则返回空字符串
        """
        if audio_path is None:
            return ""
        return self.model.recognize(audio_path)

    def is_loaded(self) -> bool:
        """检查 ASR 模型是否已加载"""
        return self.model.is_loaded()


class TTSService:
    """TTS 服务 - 对外暴露的主要功能

    封装 TTS 模型和 ASR 服务，提供完整的语音合成功能
    """

    def __init__(self, tts_model: TTSModel = None, asr_service: ASRService = None):
        self.tts_model = tts_model or TTSModel()
        self.asr_service = asr_service or ASRService()
        self._inference_lock = threading.Lock()

    def synthesize(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_len: int = 600,
        normalize: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """合成语音

        如果提供了 prompt_wav_path 但没有 prompt_text，会自动调用 ASR 识别

        Args:
            text: 目标文本
            prompt_wav_path: 参考音频路径
            prompt_text: 参考文本（如不提供将自动识别）
            cfg_value: CFG 值
            inference_timesteps: 推理时间步
            max_len: 最大生成长度
            normalize: 是否文本正则化

        Returns:
            (采样率, 音频 numpy 数组)
        """
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Text input cannot be empty")

        # 自动识别 prompt 文本
        if prompt_wav_path and not prompt_text:
            prompt_text = self.asr_service.transcribe(prompt_wav_path)

        with self._inference_lock:
            wav = self.tts_model.generate(
                text=text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_len,
                normalize=normalize,
            )

        return (self.tts_model.get_sample_rate(), wav)

    def is_tts_loaded(self) -> bool:
        """检查 TTS 模型是否已加载"""
        return self.tts_model.is_loaded()

    def is_asr_loaded(self) -> bool:
        """检查 ASR 模型是否已加载"""
        return self.asr_service.is_loaded()


# ==================== Pydantic 模型 ====================

class TTSRequest(BaseModel):
    """TTS 请求模型（兼容 IndexTTS2）"""
    text: str
    spk_audio_prompt: str = "tests/sample_prompt.wav"
    return_audio: bool = False
    # 兼容性参数（未使用）
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 0.65
    emo_text: Optional[str] = None
    emo_vec: Optional[list] = None
    emo_random: bool = False
    max_text_tokens_per_segment: int = 120
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500
    # VoxCPM 特定参数
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    max_len: int = 600
    normalize: bool = False


class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # pending, processing, completed, failed
    message: str
    result_path: Optional[str] = None


class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str
    tts_model_loaded: bool
    asr_model_loaded: bool


# ==================== API 应用 ====================

class VoxCPMAPI:
    """VoxCPM API 应用类

    封装 FastAPI 应用和路由逻辑
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir
        self.app = FastAPI(
            title="VoxCPM API",
            description="VoxCPM Text-to-Speech API (IndexTTS2 compatible)",
            version="2.0.0",
        )
        self.tasks = {}
        self._tts_service: Optional[TTSService] = None

        # 初始化目录
        self._init_directories()

        # 注册路由
        self._register_routes()

    def _init_directories(self):
        """初始化所需目录"""
        for dir_path in [APIConfig.OUTPUT_DIR, APIConfig.PROMPTS_DIR,
                         APIConfig.UPLOAD_DIR, APIConfig.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)

        # 清空历史生成音频
        for filename in os.listdir(APIConfig.OUTPUT_DIR):
            file_path = os.path.join(APIConfig.OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}', file=sys.stderr)

    def get_tts_service(self) -> TTSService:
        """获取 TTS 服务单例"""
        if self._tts_service is None:
            tts_model = TTSModel(model_dir=self.model_dir) if self.model_dir else TTSModel()
            self._tts_service = TTSService(tts_model=tts_model)
        return self._tts_service

    def _register_routes(self):
        """注册所有路由"""

        @self.app.on_event("startup")
        async def startup_event():
            """启动事件"""
            print("[VoxCPMAPI] Starting VoxCPM TTS API...", file=sys.stderr)

        @self.app.get("/")
        async def root():
            """根路径 - API 信息"""
            return {
                "service": "VoxCPM TTS API",
                "version": "2.0.0",
                "status": "running",
                "endpoints": {
                    "health": "/health",
                    "tts_create_task": "/api/v1/tts/tasks",
                    "tts_get_task": "/api/v1/tts/tasks/{task_id}",
                    "tts_get_result": "/api/v1/tts/tasks/{task_id}/result",
                    "upload_prompt": "/api/v1/tts/upload",
                }
            }

        @self.app.get("/health", response_model=HealthStatus)
        async def health():
            """健康检查"""
            service = self.get_tts_service()
            return HealthStatus(
                status="healthy",
                tts_model_loaded=service.is_tts_loaded(),
                asr_model_loaded=service.is_asr_loaded(),
            )

        @self.app.post("/api/v1/tts/tasks", response_model=TaskStatus)
        async def create_tts_task(request: TTSRequest, background_tasks: BackgroundTasks):
            """创建 TTS 任务

            支持：
            - 同步模式：return_audio=True 时直接返回音频文件
            - 异步模式：后台处理，返回任务 ID
            """
            # 检查参考音频文件是否存在
            if not os.path.exists(request.spk_audio_prompt):
                raise HTTPException(status_code=400, detail="Prompt audio file does not exist")

            task_id = str(uuid.uuid4())
            service = self.get_tts_service()

            # 同步模式：直接返回音频
            if request.return_audio:
                try:
                    output_path = os.path.join(APIConfig.OUTPUT_DIR, f"{task_id}.wav")

                    sample_rate, wav = service.synthesize(
                        text=request.text,
                        prompt_wav_path=request.spk_audio_prompt,
                        prompt_text=request.emo_text,
                        cfg_value=request.cfg_value,
                        inference_timesteps=request.inference_timesteps,
                        max_len=request.max_len,
                        normalize=request.normalize,
                    )

                    sf.write(output_path, wav, sample_rate)
                    return FileResponse(output_path, media_type='audio/wav', filename=f"{task_id}.wav")

                except Exception as e:
                    import traceback
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate audio: {str(e)}\n{traceback.format_exc()}"
                    )

            # 异步模式
            self.tasks[task_id] = {
                "status": "pending",
                "message": "Task created, waiting to be processed",
                "result_path": None
            }

            background_tasks.add_task(self._process_tts_task, task_id, request)

            return TaskStatus(
                task_id=task_id,
                status=self.tasks[task_id]["status"],
                message=self.tasks[task_id]["message"],
                result_path=self.tasks[task_id]["result_path"]
            )

        @self.app.get("/api/v1/tts/tasks", response_class=FileResponse)
        async def create_tts_task_sync(text: str, prompt_audio: str = "tests/sample_prompt.wav"):
            """GET 方式创建 TTS 任务（同步）"""
            if not os.path.exists(prompt_audio): 
                raise HTTPException(status_code=400, detail="Prompt audio file does not exist")

            task_id = str(uuid.uuid4())
            output_path = os.path.join(APIConfig.OUTPUT_DIR, f"{task_id}.wav")

            try:
                service = self.get_tts_service()

                def infer_with_timeout():
                    sample_rate, wav = service.synthesize(
                        text=text,
                        prompt_wav_path=prompt_audio,
                        prompt_text=None,
                        cfg_value=2.0,
                        inference_timesteps=10,
                        max_len=600,
                        normalize=False,
                    )
                    sf.write(output_path, wav, sample_rate)

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(infer_with_timeout)
                    future.result(timeout=APIConfig.INFERENCE_TIMEOUT)

                return FileResponse(output_path, media_type='audio/wav', filename=f"{task_id}.wav")

            except Exception as e:
                if os.path.exists(output_path):
                    os.remove(output_path)
                import traceback
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate audio: {str(e)}\n{traceback.format_exc()}"
                )

        @self.app.get("/api/v1/tts/tasks/{task_id}", response_model=TaskStatus)
        async def get_task_status(task_id: str):
            """获取任务状态"""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")

            task = self.tasks[task_id]
            return TaskStatus(
                task_id=task_id,
                status=task["status"],
                message=task["message"],
                result_path=task["result_path"]
            )

        @self.app.get("/api/v1/tts/tasks/{task_id}/result")
        async def get_task_result(task_id: str):
            """获取任务结果（音频文件）"""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")

            task = self.tasks[task_id]
            if task["status"] != "completed":
                raise HTTPException(status_code=400, detail=f"Task is not completed yet. Current status: {task['status']}")

            if not task["result_path"] or not os.path.exists(task["result_path"]):
                raise HTTPException(status_code=404, detail="Result file not found")

            return FileResponse(task["result_path"], media_type='audio/wav', filename=f"{task_id}.wav")

        @self.app.post("/api/v1/tts/upload")
        async def upload_prompt_audio(file: UploadFile = File(...)):
            """上传参考音频文件"""
            if file is None:
                raise HTTPException(status_code=400, detail="文件为空，未接收到上传内容")

            filename = f"record_{int(time.time())}_{file.filename}"
            save_path = os.path.join(APIConfig.UPLOAD_DIR, filename)

            try:
                with open(save_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

            server_path = os.path.abspath(save_path)

            return JSONResponse(
                status_code=200,
                content={
                    "code": 0,
                    "msg": "上传成功",
                    "path": server_path
                }
            )

    def _process_tts_task(self, task_id: str, request: TTSRequest):
        """后台处理 TTS 任务"""
        try:
            self.tasks[task_id]["status"] = "processing"
            self.tasks[task_id]["message"] = "Task is being processed"

            output_path = os.path.join(APIConfig.OUTPUT_DIR, f"{task_id}.wav")
            service = self.get_tts_service()

            sample_rate, wav = service.synthesize(
                text=request.text,
                prompt_wav_path=request.spk_audio_prompt,
                prompt_text=request.emo_text,
                cfg_value=request.cfg_value,
                inference_timesteps=request.inference_timesteps,
                max_len=request.max_len,
                normalize=request.normalize,
            )

            sf.write(output_path, wav, sample_rate)

            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["message"] = "Task completed successfully"
            self.tasks[task_id]["result_path"] = output_path

        except Exception as e:
            import traceback
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["message"] = f"Task failed: {str(e)}\n{traceback.format_exc()}"
            self.tasks[task_id]["result_path"] = None


# ==================== 应用入口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="VoxCPM API Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="VoxCPM model path or HuggingFace model ID",
    )
    return parser.parse_args()


def main():
    """主函数"""
    cmd_args = parse_args()

    # 创建 API 应用
    api = VoxCPMAPI(model_dir=cmd_args.model_dir)

    # 运行服务
    import uvicorn
    uvicorn.run(api.app, host=cmd_args.host, port=cmd_args.port)


if __name__ == "__main__":
    main()

"""
本地 ASR 识别器。

基于 FunASR 的本地语音识别功能。
"""
import argparse
import logging
from typing import Optional

import torch
from funasr import AutoModel

from voxcpm.utils import get_test_logger


class ASRLocalRecognizer:
    """
    基于 FunASR 的本地 ASR 识别类。
    """

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        device: Optional[str] = None,
        disable_update: bool = True,
        log_level: str = "INFO",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        初始化 ASR 识别器。

        Args:
            model_id: FunASR 模型 ID，默认为 "iic/SenseVoiceSmall"
            device: 设备（"cuda:0", "cpu" 等），如果为 None 则自动检测
            disable_update: 是否禁用模型更新
            log_level: 日志级别
            logger: 自定义 logger，如果为 None 则使用默认 logger
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.logger = logger or get_test_logger(__file__, console_output=False)

        self.logger.info("初始化 ASR 模型: %s, 设备: %s", model_id, device)
        self.asr_model = AutoModel(
            model=model_id,
            disable_update=disable_update,
            log_level=log_level,
            device=device,
        )
        self.logger.info("ASR 模型加载完成")

    def recognize(self, audio_path: Optional[str], language: str = "auto", use_itn: bool = True) -> str:
        """
        识别音频文件，返回识别文本。

        Args:
            audio_path: 音频文件路径，如果为 None 或空字符串则返回空字符串
            language: 语言设置，"auto" 表示自动检测
            use_itn: 是否使用逆文本正则化（Inverse Text Normalization）

        Returns:
            识别得到的文本字符串
        """
        if not audio_path:
            return ""

        try:
            res = self.asr_model.generate(input=audio_path, language=language, use_itn=use_itn)
            text = res[0]["text"].split("|>")[-1]
            self.logger.debug("识别结果: %s", text)
            return text
        except Exception as e:
            self.logger.error("ASR 识别失败: %s", e)
            return ""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 FunASR 的简单 ASR 识别 Demo")
    parser.add_argument("--audio-path", type=str, default="s2st_demo/input/example.wav", help="音频文件路径")
    parser.add_argument("--model-id", type=str, default="iic/SenseVoiceSmall", help="模型 ID")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_itn", action="store_true", help="是否使用逆文本正则化")
    return parser


def main():
    """
    命令行入口：单次音频识别测试。
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    asr = ASRLocalRecognizer()
    text = asr.recognize(audio_path=args.audio_path, use_itn=args.use_itn)


if __name__ == "__main__":
    main()

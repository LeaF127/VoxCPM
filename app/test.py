import gradio as gr
import numpy as np
import soundfile as sf
import librosa

def change_pitch(audio, pitch_shift=2):
    if audio is None:
        return None
    
    sr, y = audio
    # 使用librosa进行音调变换
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
    # 将float64转换为16bit整数
    y_shifted = np.clip(y_shifted, -1.0, 1.0)  # 限制在[-1, 1]范围内
    y_shifted = (y_shifted * 32767).astype(np.int16)  # 转换为16bit整数
    return (sr, y_shifted)

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 实时音调变换示例")
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            streaming=True,
            label="麦克风输入"
        )
        audio_output = gr.Audio(
            streaming=True,
            label="音调变换输出",
            autoplay=True
        )
        
        pitch_slider = gr.Slider(
            minimum=-12,
            maximum=12,
            value=2,
            step=1,
            label="音调变化（半音）"
        )
    
    # 设置流式处理
    audio_input.stream(
        fn=change_pitch,
        inputs=[audio_input, pitch_slider],
        outputs=[audio_output],
        time_limit=30,
        stream_every=0.1
    )

if __name__ == "__main__":
    demo.launch()

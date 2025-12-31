python s2st_demo/tts.py -t "你好，这是一条来自 VoxCPM 的测试语音。" --model-path ./models/openbmb__VoxCPM-0.5B/ --output ./s2st_demo/output/output.wav --prompt-wav ./examples/example.wav --prompt-text "just by listening a few minutes a day you'll be able to eliminate negative thoughts by conditioning your mind to be more positive" --normalize

python s2st_demo/asr_translate.py --audio-path ./examples/amiya.wav

python s2st_demo/run.py --audio s2st_demo/input/example.wav --model-path "./models/openbmb__VoxCPM-0.5B/" --tts-text-source trans --output s2st_demo/output/pipeline_example.wav --normalize

# 长音频—中到英
python s2st_demo/run.py \
  --ws-url "ws://175.24.179.12:9389/dotcwsasr" \
  --audio s2st_demo/input/16k/1133_tenlin.wav \
  --model-path "./models/openbmb__VoxCPM-0.5B/" \
  --tts-text-source trans \
  --output s2st_demo/output/pipeline_streaming.wav \
  --normalize \
  --streaming \
  --segment-output-dir s2st_demo/output/segments

# 短音频—中到英
python s2st_demo/run.py \
  --ws-url "ws://175.24.179.12:9389/dotcwsasr" \
  --audio s2st_demo/input/amiya_16k.wav \
  --model-path "./models/openbmb__VoxCPM-0.5B/" \
  --tts-text-source trans \
  --output s2st_demo/output/amiya.wav \
  --normalize \
  --streaming \
  --segment-output-dir s2st_demo/output/amiya

ffmpeg -i ./s2st_demo/input/44k/nasti_1s.wav -ac 1 -ar 16000 -c:a pcm_s16le ./s2st_demo/input/16k/nasti_1s_16k.wav

# 短音频—英到中
python s2st_demo/run.py \
  --lan-id 1 \
  --from-lang en \
  --to-lang zh \
  --ws-url "ws://175.24.179.12:9389/dotcwsasr" \
  --audio s2st_demo/input/16k/example_16k.wav \
  --model-path "./models/openbmb__VoxCPM-0.5B/" \
  --tts-text-source trans \
  --output s2st_demo/output/example_cn.wav \
  --normalize \
  --streaming \
  --segment-output-dir s2st_demo/output/example_cn

# 短音频（1s）—中到英
python s2st_demo/run.py \
  --ws-url "ws://175.24.179.12:9389/dotcwsasr" \
  --audio s2st_demo/input/16k/silverash_16k.wav \
  --model-path "./models/openbmb__VoxCPM-0.5B/" \
  --tts-text-source trans \
  --output s2st_demo/output/silverash.wav \
  --normalize \
  --streaming \
  --segment-output-dir s2st_demo/output/silverash
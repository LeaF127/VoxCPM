python s2st_demo/tts.py -t "你好，这是一条来自 VoxCPM 的测试语音。" --model-path ./models/openbmb__VoxCPM-0.5B/ --output ./s2st_demo/output/output.wav --prompt-wav ./examples/example.wav --prompt-text "just by listening a few minutes a day you'll be able to eliminate negative thoughts by conditioning your mind to be more positive" --normalize

python s2st_demo/asr_translate.py --audio-path ./examples/amiya.wav

python s2st_demo/run.py --audio s2st_demo/input/example.wav --model-path "./models/openbmb__VoxCPM-0.5B/" --tts-text-source trans --output s2st_demo/output/pipeline_example.wav --normalize

python s2st_demo/run.py \
  --audio s2st_demo/input/1133_tenlin.wav \
  --model-path "./models/openbmb__VoxCPM-0.5B/" \
  --tts-text-source trans \
  --output s2st_demo/output/pipeline_streaming.wav \
  --normalize \
  --streaming \
  --segment-output-dir s2st_demo/output/segments

python s2st_demo/run.py \
  --audio s2st_demo/input/jessica.wav \
  --model-path "./models/openbmb__VoxCPM-0.5B/" \
  --tts-text-source trans \
  --output s2st_demo/output/jessica.wav \
  --normalize \
  --streaming \
  --segment-output-dir s2st_demo/output/jessica
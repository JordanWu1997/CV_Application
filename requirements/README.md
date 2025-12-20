# Requirements

- This directory contains requirements for different usage (`mediapipe`, `ultralytic`, etc.)
- For now I use `requirements.txt` contains all usage cases

## mediapipe

- Since `mediapipe==0.10.20` requires `numpy<2.0.0`, it has to be installed at the last
  - `pip install 'numpy<2.0.0' mediapipe==0.10.20`
- I keep `mediapipe==0.10.20` since latest `mediapipe` removed `solutions`, `draw_utils` modules

## transformers

- `pip install git+https://github.com/huggingface/transformers.git`

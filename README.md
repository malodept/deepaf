
# DeepAF: Intelligent Autofocus (Minimal)
Colab-ready minimal code to simulate defocus, train simple CNNs to estimate distance-to-focus and classify in/out-of-focus, and run a 1–2 step control loop on ROI.

## Quickstart (Colab)
1. Open `notebooks/DeepAF.ipynb` in Colab.
2. Runtime → GPU (T4).
3. Run cells top-to-bottom. It will install deps, synthesize data, train tiny models for few epochs, and run a demo.

## Local dev
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.simulate_defocus --out data_synth --n 2000
python -m src.models.train_step_estimator --data data_synth --epochs 3
python -m src.models.train_focus_discriminator --data data_synth --epochs 3
python -m src.af.inference_controller --video assets/sample.mp4 --weights runs/step_estimator.pt runs/focus_discriminator.pt

## Structure
- src/data/simulate_defocus.py : synth generator and blur model with focus breathing
- src/models/step_estimator.py : small CNN regressor
- src/models/focus_discriminator.py : small CNN classifier
- src/models/train_step_estimator.py : trainer
- src/models/train_focus_discriminator.py : trainer
- src/af/inference_controller.py : 1–2 step loop demo on synthetic ROI
- src/roi/detector_tracker.py : YOLO wrapper (optional)
- src/eval/metrics.py : Tenengrad and Laplacian sharpness

Notes:
- For real videos, place a sample at assets/sample.mp4 and run inference_controller with --use_yolo 1 if YOLO is available.

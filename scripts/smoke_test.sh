#!/usr/bin/env bash
set -euo pipefail

python -c "from ultralytics import YOLO; import numpy as np, torch, time; model=YOLO('models/HockeyAI_model_weight.pt'); model.to('cuda:0'); frame=np.zeros((720,1280,3),dtype=np.uint8); model.predict(frame,device='cuda:0',verbose=False); torch.cuda.synchronize(); t=time.perf_counter(); model.predict(frame,device='cuda:0',verbose=False); torch.cuda.synchronize(); dt=time.perf_counter()-t; print('GPU:', next(model.model.parameters()).device); print('Inference:', round(dt,4), 'sec'); print('FPS:', round(1/dt,2))"

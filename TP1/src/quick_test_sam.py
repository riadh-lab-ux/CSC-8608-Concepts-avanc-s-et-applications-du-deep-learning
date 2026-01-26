cat > TP1/src/quick_test_sam.py <<'PY'
import numpy as np
import cv2
from pathlib import Path

from sam_utils import load_sam_predictor, predict_mask_from_box, get_device

img_path = next(Path("TP1/data/images").glob("*.jpg"))
bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# choisis le bon ckpt selon ce que tu as téléchargé
ckpt = "TP1/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

pred = load_sam_predictor(ckpt, model_type=model_type)

H, W = rgb.shape[:2]
# bbox "à la main" : un carré au centre (tu peux ajuster ensuite)
box = np.array([W*0.25, H*0.25, W*0.75, H*0.75], dtype=np.int32)

mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)
print("device", get_device())
print("img", rgb.shape, "mask", mask.shape, "score", score, "mask_sum", int(mask.sum()))
PY


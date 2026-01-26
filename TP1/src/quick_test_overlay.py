import numpy as np
import cv2
from pathlib import Path

from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


def main():
    # 1) Charger le modèle UNE SEULE FOIS
    ckpt = "TP1/models/sam_vit_h_4b8939.pth"
    pred = load_sam_predictor(ckpt, model_type="vit_h")

    # 2) Dossier de sortie
    out_dir = Path("TP1/outputs/overlays")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Boucle sur TOUTES les images jpg
    img_paths = sorted(Path("TP1/data/images").glob("*.jpg"))
    if not img_paths:
        raise SystemExit("Aucune image .jpg trouvée dans TP1/data/images")

    for img_path in img_paths:
        print(f"Processing {img_path.name}...")

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("  -> skip (cv2.imread failed)")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        
        box = np.array([50, 50, 250, 250], dtype=np.int32)

        mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)

        m_area = mask_area(mask)
        m_bbox = mask_bbox(mask)
        m_per = mask_perimeter(mask)

        overlay = render_overlay(rgb, mask, box, alpha=0.5)

        out_path = out_dir / f"overlay_{img_path.stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"{img_path.name} -> score: {score:.3f}, area: {m_area}, bbox: {m_bbox}, perimeter: {m_per:.1f}")
        print("saved:", out_path)


if __name__ == "__main__":
    main()

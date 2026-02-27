import time
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz  # <-- manquant dans l'énoncé

# Wrapper MLOps pour extraire les logits purs (requis par Captum)
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


def now_sync(device):
    """Temps fiable sur GPU : on synchronise avant de lire time.time()."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time()


# 1. Chargement de l'image via argument terminal
image_path = sys.argv[1] if len(sys.argv) > 1 else "normal_1.jpeg"
print(f"Analyse de l'image : {image_path}")
image = Image.open(image_path).convert("RGB")

# 2. Chargement du processeur et du modèle
model_name = "Aunsiels/resnet-pneumonia-detection"
processor = AutoImageProcessor.from_pretrained(model_name)
hf_model = AutoModelForImageClassification.from_pretrained(model_name)  # TODO rempli

wrapped_model = ModelWrapper(hf_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model.to(device)
wrapped_model.eval()

inputs = processor(images=image, return_tensors="pt")
input_tensor = inputs["pixel_values"].to(device)
input_tensor.requires_grad_(True)

# Warmup / cold start
_ = wrapped_model(input_tensor)

# 3. Inférence et chronométrage
start_infer = now_sync(device)
with torch.no_grad():
    logits = wrapped_model(input_tensor)
end_infer = now_sync(device)

predicted_class_idx = logits.argmax(-1).item()

print(f"Temps d'inférence : {end_infer - start_infer:.4f} secondes")
print(f"Classe prédite : {hf_model.config.id2label[predicted_class_idx]} (idx={predicted_class_idx})")

# 4. Explicabilité : Grad-CAM
# Dernière couche conv (selon l'architecture HF ResNet)
target_layer = wrapped_model.model.resnet.encoder.stages[-1].layers[-1]

start_xai = now_sync(device)

# Instancier LayerGradCam avec le modèle (wrapped) et la couche cible
layer_gradcam = LayerGradCam(wrapped_model, target_layer)  # TODO rempli

# Calculer les attributions pour la classe prédite
attributions_gradcam = layer_gradcam.attribute(input_tensor, target=predicted_class_idx)  # TODO rempli

end_xai = now_sync(device)
print(f"Temps d'explicabilité (Grad-CAM) : {end_xai - start_xai:.4f} secondes")

# 5. Visualisation
# Upsample de la carte Grad-CAM à la taille d'entrée
upsampled_attr = LayerAttribution.interpolate(attributions_gradcam, input_tensor.shape[2:])

# Image originale au format numpy (H, W, C)
H, W = input_tensor.shape[2], input_tensor.shape[3]
original_img_np = np.array(image.resize((W, H)))

# Attributions -> numpy (H, W, 1)
attr_gradcam_np = upsampled_attr.squeeze().detach().cpu().numpy()
attr_gradcam_np = np.expand_dims(attr_gradcam_np, axis=2)

fig, axis = viz.visualize_image_attr(
    attr_gradcam_np,
    original_img_np,
    method="blended_heat_map",
    sign="positive",
    show_colorbar=True,
    title=f"Grad-CAM - Pred: {hf_model.config.id2label[predicted_class_idx]}"
)

output_filename = f"gradcam_{image_path.split('.')[0]}.png"
fig.savefig(output_filename, bbox_inches="tight", dpi=200)
print(f"Visualisation sauvegardée dans {output_filename}")

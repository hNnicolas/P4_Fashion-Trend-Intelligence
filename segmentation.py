import os
import io
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Chargement des variables d'environnement
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_ID = "sayeed99/segformer_b3_clothes"

# Répertoires et constantes
IMAGES_DIR = "assets/images"
MAX_IMAGE_SIZE = 512

# Liste des classes dans l'ordre de l'API
CLASSES = [
    "background", "hat", "hair", "sunglasses", "upper-clothes", "dress",
    "coat", "socks", "pants", "gloves", "scarf", "skirt", "face",
    "left-arm", "right-arm", "left-leg", "right-leg", "left-shoe", "right-shoe"
]

# Palette de couleurs RGB pour chaque classe
PALETTE = [
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
    [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85],
    [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255], [85, 255, 170],
    [170, 255, 85], [255, 255, 0], [255, 170, 0], [255, 0, 255]
]

def load_and_resize_image(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img

def call_hf_api(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/octet-stream"
    }

    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    response = requests.post(url, headers=headers, data=image_bytes)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur API {response.status_code}: {response.text}")

def decode_mask(mask_base64):
    mask_bytes = base64.b64decode(mask_base64)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_arr = np.array(mask_img)
    mask_bin = (mask_arr > 0).astype(np.uint8)
    return mask_bin

def build_color_mask(api_result, shape):
    final_mask = np.zeros(shape, dtype=np.uint8)
    for obj in api_result:
        label_name = obj.get("label", "").lower()
        if label_name in [c.lower() for c in CLASSES]:
            label_index = [c.lower() for c in CLASSES].index(label_name)
            mask_bin = decode_mask(obj["mask"])
            final_mask[mask_bin == 1] = label_index
    return final_mask

def apply_palette(mask_array):
    color_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for label_index in np.unique(mask_array):
        if label_index < len(PALETTE):
            color_mask[mask_array == label_index] = PALETTE[label_index]
    return color_mask

def apply_mask_effect_with_selection(img, mask_array, selected_classes):
    img_np = np.array(img).copy()
    modified = img_np.copy()

    # Convertir noms en indices
    selected_indices = [CLASSES.index(c) for c in selected_classes]

    for label_index in np.unique(mask_array):
        if label_index == 0 or label_index not in selected_indices:
            continue
        
        class_mask = mask_array == label_index

        if label_index == CLASSES.index("hair"):
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray_rgb = np.stack((gray,) * 3, axis=-1)
            modified[class_mask] = gray_rgb[class_mask]

        elif label_index == CLASSES.index("coat"):
            modified[class_mask] = [255, 0, 0]

        elif label_index == CLASSES.index("pants"):
            modified[class_mask] = [0, 255, 0]

        elif label_index == CLASSES.index("upper-clothes"):
            upper = img_np[class_mask]
            saturated = np.clip(upper * 1.5, 0, 255).astype(np.uint8)
            modified[class_mask] = saturated

    return modified


def show_and_save_panel(img, color_mask, modified_img, detected_labels, image_name):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(img)
    axs[0].set_title("Image originale")
    axs[0].axis("off")

    axs[1].imshow(color_mask)
    axs[1].set_title("Masque colorisé")
    axs[1].axis("off")

    axs[2].imshow(modified_img)
    axs[2].set_title("Image modifiée (désaturation vêtements)")
    axs[2].axis("off")

    # Affiche la légende à droite
    legend_text = "\n".join(detected_labels)
    plt.figtext(0.92, 0.5, legend_text, fontsize=10, va='center')

    plt.tight_layout()
    plt.show()  # Affichage bloquant pour interaction

    # Demander à l'utilisateur s’il souhaite sauvegarder
    user_input = input("💾 Souhaitez-vous sauvegarder ce panneau ? (o/n) : ").strip().lower()
    if user_input in ['o', 'y', 'oui', 'yes']:
        os.makedirs("outputs", exist_ok=True)
        save_path = os.path.join("outputs", f"panel_{image_name}")
        fig.savefig(save_path)
        print(f"✅ Panneau sauvegardé dans {save_path}")
    else:
        print("❌ Panneau non sauvegardé.")

    plt.close()

def main():
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".png")]
    if not images:
        print("❌ Aucune image PNG trouvée dans", IMAGES_DIR)
        return

    selected_classes = ["hair", "coat", "pants", "upper-clothes"]

    for image_name in images:
        print(f"\n📸 Traitement de : {image_name}")
        img_path = os.path.join(IMAGES_DIR, image_name)
        img = load_and_resize_image(img_path)

        try:
            result = call_hf_api(img)

            if isinstance(result, list) and len(result) > 0 and "mask" in result[0]:
                final_mask = build_color_mask(result, (img.height, img.width))
                color_mask = apply_palette(final_mask)

                detected_labels = [obj["label"] for obj in result]
                modified_img = apply_mask_effect_with_selection(img, final_mask, selected_classes)

                show_and_save_panel(img, color_mask, modified_img, detected_labels, image_name)

            else:
                print("Réponse API inattendue :", result)

        except Exception as e:
            print("❌ Erreur:", e)
            
if __name__ == "__main__":
    main()


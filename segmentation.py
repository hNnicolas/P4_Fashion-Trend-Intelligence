import os
import io
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_ID = "sayeed99/segformer_b3_clothes"

IMAGES_DIR = "assets/images"
MAX_IMAGE_SIZE = 512

# Liste des classes dans l'ordre correspondant à l'API
CLASSES = [
    "background", "hat", "hair", "sunglasses", "upper-clothes", "dress",
    "coat", "socks", "pants", "gloves", "scarf", "skirt", "face",
    "left-arm", "right-arm", "left-leg", "right-leg", "left-shoe", "right-shoe"
]

# Palette RGB par classe
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
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")  # niveaux de gris
    mask_arr = np.array(mask_img)
    # Convertir pixels non nuls en 1 (masque binaire)
    mask_bin = (mask_arr > 0).astype(np.uint8)
    return mask_bin

def build_color_mask(api_result, shape):
    # Initialiser le masque final avec des zéros (background)
    final_mask = np.zeros(shape, dtype=np.uint8)
    for obj in api_result:
        label_name = obj.get("label", "").lower()
        if label_name in [c.lower() for c in CLASSES]:
            label_index = [c.lower() for c in CLASSES].index(label_name)
            mask_bin = decode_mask(obj["mask"])
            # On applique le label_index partout où le masque binaire est à 1
            final_mask[mask_bin == 1] = label_index
    return final_mask

def apply_palette(mask_array):
    color_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for label_index in np.unique(mask_array):
        if label_index < len(PALETTE):
            color_mask[mask_array == label_index] = PALETTE[label_index]
    return color_mask

def show_segmentation(image, color_mask, detected_labels):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Image originale")
    axs[0].axis("off")

    axs[1].imshow(color_mask)
    axs[1].set_title("Masque colorisé")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()  

    print("\n🧵 Légende détectée :")
    for label in detected_labels:
        print(f"  - {label}")

def main():
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".png")]
    if not images:
        print("❌ Aucune image PNG trouvée dans", IMAGES_DIR)
        return

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
                show_segmentation(img, color_mask, detected_labels)

            else:
                print("Réponse API inattendue :", result)

        except Exception as e:
            print("❌ Erreur:", e)

if __name__ == "__main__":
    main()

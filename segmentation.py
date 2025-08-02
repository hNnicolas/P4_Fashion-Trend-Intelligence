import os
import io
import json
import base64
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---

# Dossier des images et annotations
IMAGES_DIR = "assets/images"
ANNOTATIONS_FILE = "assets/images/annotations.json"

# Modèle HF à utiliser
MODEL_ID = "sayeed99/segformer_b3_clothes"

# Taille max des images pour optimiser la requête
MAX_IMAGE_SIZE = 512

# --- Fonctions utilitaires ---

def load_and_resize_image(path, max_size=MAX_IMAGE_SIZE):
    """Charge une image, la convertit en RGB et la redimensionne."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img

def call_hf_api(image: Image.Image, api_token: str):
    """
    Envoie une image à l'API Hugging Face et récupère la segmentation.
    """
    # Convertir l'image en bytes PNG
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/octet-stream"
    }
    
    api_url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

    response = requests.post(api_url, headers=headers, data=image_bytes)
    
    # Vérification du statut HTTP
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erreur HTTP {response.status_code} : {response.text}")

def display_segmentation_mask(mask_base64):
    """Décode et affiche le masque de segmentation encodé en base64."""
    # Décoder la chaîne base64 en bytes
    mask_bytes = base64.b64decode(mask_base64)
    # Charger l'image depuis ces bytes
    mask_img = Image.open(io.BytesIO(mask_bytes))
    # Afficher l'image du masque
    plt.imshow(mask_img)
    plt.title("Carte de segmentation")
    plt.axis("off")
    plt.show()

# --- Script principal ---

def main():
    print(f"Répertoire courant : {os.getcwd()}")

    # Vérification des images
    if not os.path.isdir(IMAGES_DIR):
        print(f"Erreur : dossier '{IMAGES_DIR}' introuvable.")
        return
    
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]
    if not images:
        print("Aucune image PNG trouvée dans le dossier.")
        return
    
    print(f"Images trouvées ({len(images)}):")
    for img in images:
        print(" -", img)

    # Chargement annotations (optionnel)
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, "r") as f:
            annotations = json.load(f)
        print("Annotations chargées.")
    else:
        annotations = {}
        print("Aucune annotation chargée.")
    
    # Prendre la première image
    first_img = images[0]
    img_path = os.path.join(IMAGES_DIR, first_img)

    # Charger et afficher l'image
    img = load_and_resize_image(img_path)
    plt.imshow(img)
    plt.title(f"Image : {first_img}")
    plt.axis("off")
    plt.show()

    # Afficher annotations si présentes
    if first_img in annotations:
        print("Annotations :", annotations[first_img])
    else:
        print("Aucune annotation pour cette image.")

    # Récupérer token API depuis variable d'environnement
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")

    if not api_token:
        print("Erreur : variable HUGGINGFACE_API_TOKEN non définie ou vide")
        return
    print(f"Token lu (10 premiers caractères) : {api_token[:10]}")  # affiche partiellement le token

    print(f"Envoi de l'image '{img_path}' à l'API...")

    # Appel API Hugging Face
    try:
        result = call_hf_api(img, api_token)
        print("Résultat reçu de l'API.")
        # L'API renvoie une liste avec un ou plusieurs résultats, on prend le premier
        if isinstance(result, list) and "mask" in result[0]:
            display_segmentation_mask(result[0]["mask"])
        else:
            print(json.dumps(result, indent=2))
    except Exception as e:
        print("Erreur lors de l'appel API :", e)

if __name__ == "__main__":
    main()

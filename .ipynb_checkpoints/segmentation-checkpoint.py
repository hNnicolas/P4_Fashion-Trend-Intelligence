import os
import requests
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Charger les variables d'environnement depuis .env
load_dotenv()

API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/nvidia/segformer-b3-finetuned-ade-512-512"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def segment_image(image_path):
    """
    Envoie une image redimensionnée à l'API Hugging Face segformer
    et retourne le masque de segmentation.
    """
    with Image.open(image_path) as img:
        img.thumbnail((512, 512))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    response = requests.post(API_URL, headers=headers, data=image_bytes)
    if response.status_code != 200:
        raise Exception(f"Erreur API : {response.status_code} - {response.text}")
    return response.content

def display_result(image_path, mask_bytes):
    """
    Affiche côte à côte l'image originale et son masque de segmentation.
    """
    original = Image.open(image_path)
    mask = Image.open(BytesIO(mask_bytes))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmentation")
    plt.imshow(mask)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    folder_path = "assets/images"
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            print(f"Traitement de l'image : {filename}")

            try:
                mask = segment_image(img_path)

                # Sauvegarder le masque
                output_path = os.path.join(output_folder, f"mask_{filename}")
                with open(output_path, "wb") as f:
                    f.write(mask)

                display_result(img_path, mask)

            except Exception as e:
                print(f"Erreur avec {filename} : {e}")

import os
import json
from PIL import Image
import matplotlib.pyplot as plt

images_dir = "assets/images"
annotations_file = "assets/annotations.json"

images = [f for f in os.listdir(images_dir) if f.endswith(".png")]
print(f"Images trouvées ({len(images)}):")
for img in images:
    print(" -", img)

if os.path.exists(annotations_file):
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
else:
    print("Fichier annotations.json non trouvé, aucune annotation chargée.")
    annotations = {}

if images:
    first_img = images[0]
    img_path = os.path.join(images_dir, first_img)

    img = Image.open(img_path)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Image : {first_img}")
    plt.show()

    if first_img in annotations:
        print("Annotations :", annotations[first_img])
    else:
        print("Aucune annotation trouvée pour cette image.")
else:
    print("Aucune image trouvée dans le dossier.")

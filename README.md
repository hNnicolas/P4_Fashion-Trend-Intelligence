# Fashion Trend Intelligence

---

## Présentation du projet

Bonjour,

Merci pour ton attention lors de la présentation du projet **Fashion Trend Intelligence**. Je voulais, dans ce document, te faire un récapitulatif de la mission.

Comme tu l'as compris, l'objectif global est de concevoir, développer et déployer un système d'analyse disposant des fonctionnalités suivantes :

- **Segmentation vestimentaire** : être capable d'identifier et d'isoler avec précision chaque pièce vestimentaire dans une image.
- **Analyse stylistique** : classifier les pièces selon leur nature, couleur, texture et style.
- **Agrégation de tendances** : compiler ces données sur des milliers de publications pour identifier les tendances émergentes.

Comme tu peux le voir, ce projet est ambitieux et présente plusieurs challenges !  
Mais ne t'inquiète pas, ton rôle sera de travailler sur la première fonctionnalité, la **Segmentation vestimentaire**.

---

## Fonctionnalité de segmentation

Pour cette fonctionnalité, la première étape est d'utiliser un modèle pré-entraîné capable d'identifier les différentes pièces vestimentaires dans une photo, comme illustré ci-dessous.

![Segmentation vestimentaire](assets/images/screenshot_segmentation.png)

*(Remplace le chemin par celui de ton screenshot dans le dossier du projet)*

---

## Technologies utilisées

- Python 3
- API Hugging Face avec le modèle `sayeed99/segformer_b3_clothes`
- Bibliothèques : `Pillow`, `requests`, `matplotlib`, `numpy`

---

## Utilisation

1. Place tes images dans le dossier `assets/images`.
2. Défini ta variable d'environnement `HUGGINGFACE_API_TOKEN` avec ton token Hugging Face.
3. Lance le script :

```bash
python segmentation.py

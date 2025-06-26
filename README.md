# Voyageur de commerce - Algorithme génétique

---

## Prérequis

- Python **3.10**
- Pip

---

## Installation

```bash
git clone <repo_url>
cd <repo>
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt
```

## Lancement

```bash
python3 main.py
```
Exemple avec options
```bash
python3 main.py --taille_circuit 40 --taille_pop 20 --generations 1000
```
**Résultats dans le fichier ./images/ !**

## Options
| Option               | Description                                   | Par défaut          |
|----------------------|-----------------------------------------------|----------------------|
| `--csv`              | Chemin vers le fichier CSV                    | `worldcities.csv`    |
| `--taille_circuit`   | Nombre de villes dans le circuit              | `25`                 |
| `--taille_pop`       | Taille de la population génétique             | `10`                 |
| `--generations`      | Nombre de générations                         | `500`                |
| `--seed`             | Graine pour l’aléatoire                       | `712`                |

## Autre

Les labs 1 et 4 sont disponibles dans le dossier ./lab1-4 sous forme de notebooks Jupyter

## Données & Ressources

- https://cours-info.iut-bm.univ-fcomte.fr/upload/supports/S6/TechOpt/2024-2025/cours-P5.pdf  
- https://simplemaps.com/data/world-cities

Noah NICOLAS - IUT Nord Franche Comté - BUT3 ALT – 2025

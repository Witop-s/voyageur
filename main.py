import csv
import inspect
import os
import random
import logging
import math
import argparse
from geopy.distance import geodesic
from typing import List, Callable, Optional
from dataclasses import dataclass

from matplotlib import pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def log(msg: str, *args):
    frame = inspect.currentframe()
    if frame is None:
        logging.debug(msg, *args)
        return

    current = frame.f_back
    caller = current.f_back if current else None

    current_name = current.f_code.co_name if current else "?"
    caller_name = caller.f_code.co_name if caller else "?"
    lineno = current.f_lineno if current else 0

    GREEN = "\033[92m"
    WHITE = "\033[97m"

    prefix = f"{GREEN}[{lineno:>2} - {caller_name:<10}→ {current_name:<10}]{WHITE} "
    logging.debug(prefix + msg, *args)

@dataclass
class Ville:
    nom: str
    latitude: float
    longitude: float
    pays: str

def charger_villes_csv(path: str) -> List[Ville]:
    villes = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for ligne in reader:
            nom = ligne['city_ascii'] or ligne['city']
            lat = float(ligne['lat'])
            lng = float(ligne['lng'])
            pays = ligne['country']
            villes.append(Ville(nom, lat, lng, pays))
    log("Chargement de %d villes depuis %s", len(villes), path)
    return villes

def generer_circuit_aleatoire(villes: List[Ville], taille: int) -> List[Ville]:
    if taille > len(villes):
        raise ValueError("Taille du circuit supérieure au nombre de villes disponibles.")
    circuit = random.sample(villes, taille)
    log("Génération d’un circuit de %d villes : %s", taille, [v.nom for v in circuit])
    return circuit

def distance_flat(ville1: Ville, ville2: Ville) -> float:
    lat1, lon1 = ville1.latitude, ville1.longitude
    lat2, lon2 = ville2.latitude, ville2.longitude
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def distance_globe(ville1: Ville, ville2: Ville) -> float:
    coord1 = (ville1.latitude, ville1.longitude)
    coord2 = (ville2.latitude, ville2.longitude)
    return geodesic(coord1, coord2).km

def fitness(circuit: List[Ville], distance_func) -> float:
    total = 0.0
    n = len(circuit)
    for i in range(n):
        ville_src = circuit[i]
        ville_dest = circuit[(i + 1) % n]
        total += distance_func(ville_src, ville_dest)
    return total

def evaluation(population: List[List[Ville]], distance_func: Callable[[Ville, Ville], float]) -> List[float]:
    return [fitness(circuit, distance_func) for circuit in population]

def croisement(parent1: List[Ville], parent2: List[Ville]) -> List[Ville]:
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))

    log("P1: %s", [v.nom for v in parent1])
    log("P2: %s", [v.nom for v in parent2])
    log("Segment: i%d-%d", start, end)

    enfant: List[Optional[Ville]] = [None] * n
    enfant[start:end + 1] = parent1[start:end + 1]
    log("Segment: %s", [v.nom if v else "None" for v in enfant])

    p2_index = 0
    for i in range(n):
        if enfant[i] is None:
            while parent2[p2_index] in enfant:
                log("Skipping %s", parent2[p2_index].nom)
                p2_index += 1
            enfant[i] = parent2[p2_index]
            log("%s -> %d", parent2[p2_index].nom, i)
            p2_index += 1

    log("Child: %s", [v.nom for v in enfant])
    return [v for v in enfant if v is not None]

def mutation(circuit: List[Ville], prob: float = 0.1) -> List[Ville]:
    altered_c = circuit.copy()
    if random.random() < prob:
        i, j = random.sample(range(len(circuit)), 2)
        log("Mutation: %s <-> %s", altered_c[i].nom, altered_c[j].nom)
        altered_c[i], altered_c[j] = altered_c[j], altered_c[i]
    return altered_c

def selection(population: List[List[Ville]], fitnesses: List[float], taille_tournoi: int, nb_selectionnes: int) -> List[List[Ville]]:
    selectionnes = []
    for _ in range(nb_selectionnes):
        candidats = random.sample(list(zip(population, fitnesses)), taille_tournoi)
        meilleur = min(candidats, key=lambda x: x[1])
        selectionnes.append(meilleur[0])
        log("Tournoi: %s -> sélectionné: %s", [f"{[v.nom for v in c[0]]}: {c[1]}" for c in candidats], [v.nom for v in meilleur[0]])
    return selectionnes

def evolution(villes: List[Ville], taille_pop: int = 10, nb_generations: int = 50,
        prob_croisement: float = 0.9, prob_mutation: float = 0.1, taille_tournoi: int = 3
    ) -> List[Ville]:
    # Génération initiale (population aléatoire)
    population = [random.sample(villes, len(villes)) for _ in range(taille_pop)]
    fitnesses = evaluation(population, distance_globe)

    meilleur_fitness = min(fitnesses)
    meilleur_individu = population[fitnesses.index(meilleur_fitness)]

    historique_fitness = [meilleur_fitness]
    historique_moyenne = [sum(fitnesses) / len(fitnesses)]
    historique_pire = [max(fitnesses)]

    for generation in range(nb_generations):
        logging.info("=== Génération %d ===", generation)

        nouvelle_population = []

        # Génération des enfants
        while len(nouvelle_population) < taille_pop:
            # Sélection de deux parents
            parents = selection(population, fitnesses, taille_tournoi, 2)

            # Croisement
            if random.random() < prob_croisement:
                child = croisement(parents[0], parents[1])
            else:
                child = parents[0].copy()

            # Mutation
            child = mutation(child, prob_mutation)

            nouvelle_population.append(child)

        # Mise à jour de la population
        population = nouvelle_population
        fitnesses = evaluation(population, distance_globe)

        # Affichage du meilleur
        meilleur_fitness = min(fitnesses)
        meilleur_individu = population[fitnesses.index(meilleur_fitness)]

        historique_fitness.append(meilleur_fitness)
        historique_moyenne.append(sum(fitnesses) / len(fitnesses))
        historique_pire.append(max(fitnesses))

        log("Meilleur score: %.2f | Circuit: %s", meilleur_fitness, [v.nom for v in meilleur_individu])

    plt.figure(figsize=(10, 5))
    plt.plot(historique_fitness, marker='o', label="Meilleur")
    # plt.plot(historique_moyenne, marker='s', label="Moyenne")
    # plt.plot(historique_pire, marker='x', label="Pire")

    plt.title("Évolution des scores par génération")
    plt.xlabel("Génération")
    plt.ylabel("Distance (fitness)")
    plt.grid(True)
    plt.legend()
    # plt.tight_layout()
    # plt.show()
    plt.savefig("images/fitness_plot.png")
    plt.close()

    return meilleur_individu

def afficher_circuit(circuit: List[Ville]):
    # Coordonnées
    lats = [v.latitude for v in circuit] + [circuit[0].latitude]
    lngs = [v.longitude for v in circuit] + [circuit[0].longitude]
    noms = [v.nom for v in circuit]

    # Tracé du circuit
    plt.figure(figsize=(8, 6))
    plt.plot(lngs, lats, marker='o', linestyle='-', color='blue')

    # Affichage des noms des villes
    for v in circuit:
        plt.text(v.longitude + 0.1, v.latitude + 0.1, v.nom, fontsize=9)

    # Boucle de retour
    plt.plot([lngs[-2], lngs[-1]], [lats[-2], lats[-1]], color='blue', linestyle='--')

    plt.title("Meilleur circuit trouvé")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("images/graph.png")
    plt.close()

# ----------- TESTS ------------

"""
random.seed(712)

villes_importees = charger_villes_csv("worldcities.csv")

c1 = generer_circuit_aleatoire(villes_importees, taille=20)
c2 = generer_circuit_aleatoire(villes_importees, taille=20)
c3 = generer_circuit_aleatoire(villes_importees, taille=20)

log("c1: %.1f", fitness(c1, distance_globe))
log("c2: %.1f", fitness(c2, distance_globe))
log("c3: %.1f", fitness(c3, distance_globe))

population = [c1, c2, c3]
scores = evaluation(population, distance_globe)
log("Fitness population: %s", scores)

enfant = croisement(c1, c2)
log("Résultat croisement: %s", [v.nom for v in enfant])

mutant = mutation(enfant, prob=1.0)
log("Résultat mutation: %s", [v.nom for v in mutant])

selected = selection(population, scores, taille_tournoi=2, nb_selectionnes=4)

c4 = generer_circuit_aleatoire(villes_importees, taille=25)
meilleur = evolution(c4, taille_pop=10, nb_generations=500)
log("Circuit final: %s", [v.nom for v in meilleur])
afficher_circuit(meilleur)
"""

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser(description="Résolution du TSP par algorithme génétique")
    parser.add_argument("--csv", type=str, default="worldcities.csv", help="Chemin vers le fichier CSV")
    parser.add_argument("--taille_circuit", type=int, default=25, help="Nombre de villes dans le circuit")
    parser.add_argument("--taille_pop", type=int, default=10, help="Taille de la population")
    parser.add_argument("--generations", type=int, default=500, help="Nombre de générations")
    parser.add_argument("--seed", type=int, default=712, help="Graine aléatoire")
    args = parser.parse_args()

    random.seed(args.seed)

    villes_importees = charger_villes_csv(args.csv)
    circuit_depart = generer_circuit_aleatoire(villes_importees, taille=args.taille_circuit)

    meilleur = evolution(
        circuit_depart,
        taille_pop=args.taille_pop,
        nb_generations=args.generations
    )

    log("Circuit final: %s", [v.nom for v in meilleur])
    afficher_circuit(meilleur)

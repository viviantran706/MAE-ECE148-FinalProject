from collections import deque
CONSTANTE_DISTANCE = 10
W_PERS  = 10000 
W_TEMP  = 10     
W_NEIGH  = 5      
W_DELTA = 50   
W_DIST  = 20    

class Summit:
    def __init__(self, id, temperature, a_personne, voisins_ids):
        self.id = id
        self.temperature = temperature
        self.a_personne = a_personne
        self.voisins_ids = voisins_ids
        self.temp_precedente = temperature
        self.delta_t = 0

def distance_bfs(depart_id, cible_id, table_sommets):
    if depart_id == cible_id:
        return 0
    file = deque([(depart_id, 0)])
    visites = {depart_id}
    map_sommets = {s.id: s for s in table_sommets}
    while file:
        id_courant, sauts = file.popleft()
        if id_courant == cible_id:
            return sauts
        sommet = map_sommets.get(id_courant)
        if not sommet: 
            continue
        for v_id in sommet.voisins_ids:
            if v_id not in visites:
                visites.add(v_id)
                file.append((v_id, sauts + 1))
    return 999 # in case there is a summit where we can't go to 

def choisir_meilleure_cible(position_actuelle_id, table_sommets):
    meilleur_score = -float('inf')
    meilleure_cible = None
    map_sommets = {s.id: s for s in table_sommets}
    for candidat in table_sommets:
        if candidat.temperature <= 0: 
            continue
        temp_neighbor = 0
        for v_id in candidat.voisins_ids:
            if v_id in map_sommets:
                temp_voisins += map_sommets[v_id].temperature
        score_urgence = (W_PERS * candidat.a_personne)+(W_TEMP * candidat.temperature)+(W_NEIGH * temp_neighbor)+(W_DELTA * candidat.delta_t)
        nb_sauts = distance_bfs(position_actuelle_id, candidat.id, table_sommets)
        dist_reelle = nb_sauts * CONSTANTE_DISTANCE
        penalite_distance = W_DIST * dist_reelle
        final_score = score_urgence - penalite_distance
        print(f"window {candidat.id} -> score: {final_score}")
        if final_score > meilleur_score:
            meilleur_score = final_score
            meilleure_cible = candidat

    return meilleure_cible
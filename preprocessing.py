import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import joblib
import os

def build_dynamic_field_dataset(input_path, output_csv):
    print(f"Lettura file: {input_path}...")
    gdf = gpd.read_file(input_path)
    
    # Riconversione CRS per calcoli metrici (EPSG:32633)
    if gdf.crs != "EPSG:32633":
        gdf = gdf.to_crs(epsg=32633)

    # --- VERIFICA COLONNE ---
    # Definiamo le 9 variabili prodotte dallo script QGIS
    features_base = [
        "psi_el", "psi_em", "psi_hl", "psi_hm", "psi_ml", "psi_mm", 
        "w_edu", "w_heal", "w_mob"
    ]
    target_col = "ip_dyn"

    # Controllo se le colonne esistono
    missing_cols = [c for c in features_base if c not in gdf.columns]
    if missing_cols:
        print(f"ERRORE: Colonne mancanti nel file: {missing_cols}")
        print(f"Colonne disponibili: {list(gdf.columns)}")
        return

    # Estrazione coordinate centroidi
    coords = np.array(list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y)))
    tree = KDTree(coords)
    
    # Definizione Raggi di Influenza (Aloni)
    R_LOCAL = 1500   # 15 min piedi
    R_MACRO = 20000  # 60 min auto (scala territoriale)

    def fuzzy_weight(dist, t0, stiff):
        # t0 in minuti, trasformiamo distanza in proxy temporale (dist/100)
        return 1 / (1 + np.exp(stiff * (dist/100 - t0)))

    dataset = []
    total = len(gdf)
    print(f"Elaborazione di {total} neuroni territoriali...")

    for i in range(total):
        p_target = coords[i]
        row = gdf.iloc[i]
        
        def get_field_signal(radius, t0, stiff):
            indices = tree.query_ball_point(p_target, radius)
            if not indices: return np.zeros(len(features_base))
            
            nb_data = gdf.iloc[indices][features_base].values
            nb_coords = coords[indices]
            dists = np.linalg.norm(nb_coords - p_target, axis=1)
            
            weights = fuzzy_weight(dists, t0, stiff)
            w_sum = np.sum(weights)
            
            if w_sum > 0:
                return np.sum(nb_data.T * weights, axis=1) / w_sum
            return np.zeros(len(features_base))

        # COSTRUZIONE VETTORE (9 Self + 9 Local + 9 Macro = 27 variabili)
        v_self = row[features_base].values.tolist()
        v_local = get_field_signal(R_LOCAL, 15, 0.3).tolist()
        v_macro = get_field_signal(R_MACRO, 60, 0.1).tolist()
        
        full_vector = v_self + v_local + v_macro + [row[target_col]]
        dataset.append(full_vector)
        
        if i % 500 == 0: print(f"Progresso: {int((i/total)*100)}%")

    # Salvataggio
    col_names = [f"s_{c}" for c in features_base] + \
                [f"l_{c}" for c in features_base] + \
                [f"m_{c}" for c in features_base] + ["target_ip"]
    
    pd.DataFrame(dataset, columns=col_names).to_csv(output_csv, index=False)
    print(f"Preprocessing completato! File creato: {output_csv}")

# ESECUZIONE (Sostituisci col tuo file)
# build_dynamic_field_dataset("risultati_prossimita.geojson", "dataset_27var_field.csv")


build_dynamic_field_dataset("./output_elab/roma_subiaco_v3.geojson", "./output_elab/roma_subiaco_v3.csv")

#build_neural_dataset("./output_elab/roma_subiaco.geojson", "./output_elab/roma_subiaco.csv")

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, 
                       QgsProcessingParameterFeatureSource, 
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterString,
                       QgsProcessingParameterNumber,
                       QgsFeature, QgsField, QgsProcessingException)
import psycopg2
import numpy as np
import math

class ProximityEngineFuzzyEdgeToEdge(QgsProcessingAlgorithm):
    # Definizione costanti per i parametri
    INPUT_GRID = 'INPUT_GRID'
    LAYER_E_L = 'LAYER_E_L'; LAYER_E_M = 'LAYER_E_M'
    LAYER_H_L = 'LAYER_H_L'; LAYER_H_M = 'LAYER_H_M'
    LAYER_M_L = 'LAYER_M_L'; LAYER_M_M = 'LAYER_M_M'
    DB_CONN = 'DB_CONN'
    EDGES_TABLE = 'EDGES_TABLE'
    GEOM_COL = 'GEOM_COL'
    TOLERANCE = 'TOLERANCE'
    OUTPUT = 'OUTPUT'

    def name(self): return 'proximity_3pillars_fuzzy_edge'
    def displayName(self): return 'SNAI: Prossimità Fuzzy 3 Pilastri (Edge-to-Edge)'
    def group(self): return 'Analisi Territoriale'
    def groupId(self): return 'prossimita'
    def createInstance(self): return ProximityEngineFuzzyEdgeToEdge()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_GRID, 'Griglia Esagonale (32633) con w_edu, w_heal, w_mob'))
        # I 6 Layer Poligonali di Input
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_E_L, 'Istruzione Local (Poly - 15m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_E_M, 'Istruzione Macro (Poly - 60m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_H_L, 'Salute Local (Poly - 15m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_H_M, 'Salute Macro (Poly - 60m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_M_L, 'Mobilità Local (Poly - 15m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_M_M, 'Mobilità Macro (Poly - 60m)'))
        
        self.addParameter(QgsProcessingParameterNumber(self.TOLERANCE, 'Snapping Tolerance (m)', defaultValue=1000))
        self.addParameter(QgsProcessingParameterString(self.DB_CONN, 'Connessione PostgreSQL', defaultValue="host=localhost dbname=osm user=postgres password=pwd"))
        self.addParameter(QgsProcessingParameterString(self.EDGES_TABLE, 'Tabella Archi (ways)', defaultValue="ways"))
        self.addParameter(QgsProcessingParameterString(self.GEOM_COL, 'Colonna Geom Vertici DB', defaultValue="the_geom"))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Indice Prossimità Fuzzy Finale'))

    def processAlgorithm(self, parameters, context, feedback):
        grid_source = self.parameterAsSource(parameters, self.INPUT_GRID, context)
        conn = psycopg2.connect(self.parameterAsString(parameters, self.DB_CONN, context))
        cur = conn.cursor()
        edges = self.parameterAsString(parameters, self.EDGES_TABLE, context)
        g_col = self.parameterAsString(parameters, self.GEOM_COL, context)
        tol = self.parameterAsDouble(parameters, self.TOLERANCE, context)

        # --- 1. FUNZIONE LOGICA: SNAPPING AL PUNTO DI INTERSEZIONE RETE-CONFINE ---
        def snap_boundary(layer_param, label):
            feedback.pushInfo(f"Analisi confine per: {label}...")
            source = self.parameterAsSource(parameters, layer_param, context)
            if not source: return [], {}
            mapping = {}
            for f in source.getFeatures():
                wkt = f.geometry().asWkt()
                sql = f"""
                    SELECT v.id FROM {edges}_vertices_pgr v
                    JOIN {edges} e ON (v.id = e.source OR v.id = e.target)
                    WHERE ST_Intersects(e.the_geom, ST_Transform(ST_Boundary(ST_GeomFromText('{wkt}', 32633)), 4326))
                    ORDER BY ST_Transform(v.{g_col}, 32633) <-> ST_GeomFromText('{wkt}', 32633) LIMIT 1
                """
                cur.execute(sql)
                res = cur.fetchone()
                if res: mapping[f.id()] = res[0]
            return list(set(mapping.values())), mapping

        # --- 2. ESECUZIONE SNAPPING MASSIVO (0-40%) ---
        grid_feats = {f.id(): f for f in grid_source.getFeatures()}
        unique_grid_nodes, grid_map = snap_boundary(self.INPUT_GRID, "Griglia Esagonale")
        
        poi_layers = {'el': self.LAYER_E_L, 'em': self.LAYER_E_M, 'hl': self.LAYER_H_L, 'hm': self.LAYER_H_M, 'ml': self.LAYER_M_L, 'mm': self.LAYER_M_M}
        poi_data = {}
        for k, lp in poi_layers.items():
            poi_data[k], _ = snap_boundary(lp, k)
        feedback.setProgress(40)

        # --- 3. VERIFICA INTERSEZIONE DIRETTA (COSTO 0) ---
        direct_hits = {k: set() for k in poi_layers.keys()}
        for k, lp in poi_layers.items():
            source = self.parameterAsSource(parameters, lp, context)
            if not source: continue
            # Nota: per grandi dataset usare Join Spaziale pre-calcolato
            for f_poi in source.getFeatures():
                for f_grid in grid_source.getFeatures():
                    if f_grid.geometry().intersects(f_poi.geometry()):
                        direct_hits[k].add(f_grid.id())

        # --- 4. ROUTING MATRICIALE PGROUTING (40-80%) ---
        costs = {}
        cfgs = {
            'el': ('cost_walk', 15, 0.3), 'em': ('cost_car', 60, 0.1),
            'hl': ('cost_walk', 15, 0.3), 'hm': ('cost_car', 60, 0.1),
            'ml': ('cost_walk', 15, 0.3), 'mm': ('cost_car', 60, 0.1)
        }
        for k, (c_col, t_max, stiff) in cfgs.items():
            if not poi_data[k]: costs[k] = {}; continue
            sql_route = f"SELECT start_vid, min(agg_cost) FROM pgr_dijkstraCost('SELECT id, source, target, {c_col} as cost FROM {edges}', ARRAY{unique_grid_nodes}, ARRAY{poi_data[k]}, directed := false) GROUP BY start_vid"
            cur.execute(sql_route)
            costs[k] = dict(cur.fetchall())
        feedback.setProgress(80)

        # --- 5. CALCOLO FUZZY E GENERAZIONE OUTPUT (80-100%) ---
        out_fields = grid_source.fields()
        for m in ["psi_edu", "psi_heal", "psi_mob", "ip_dinamico"]: out_fields.append(QgsField(m, QVariant.Double))
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, grid_source.wkbType(), grid_source.sourceCrs())

        def fuzzy_decay(sec, t0_min, stiffness):
            if sec == 0: return 1.0
            if sec is None: return 0.0
            t = sec / 60.0
            # Funzione Sigmoide: 1 / (1 + exp(k * (t - t0)))
            return 1 / (1 + math.exp(stiffness * (t - t0_min)))

        for i, f in enumerate(grid_feats.values()):
            nid = grid_map.get(f.id())
            sc = {}
            for k, (c_col, t_max, stiff) in cfgs.items():
                if f.id() in direct_hits[k]:
                    sc[k] = 1.0
                else:
                    c_sec = costs[k].get(nid) if nid else None
                    sc[k] = fuzzy_decay(c_sec, t_max, stiff)

            # Aggregazione nei 3 Indicatori Tematici
            psi_edu = (sc['el'] * 0.6) + (sc['em'] * 0.4)
            psi_heal = (sc['hl'] * 0.4) + (sc['hm'] * 0.6)
            psi_mob = (sc['ml'] * 0.5) + (sc['mm'] * 0.5)

            # Normalizzazione Pesi Dinamici dalla Griglia
            w = np.array([f['w_edu'] or 0, f['w_heal'] or 0, f['w_mob'] or 0], dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.array([0.333, 0.333, 0.334])

            # Calcolo IP Dinamico Finale
            ip_dyn = (psi_edu * w[0]) + (psi_heal * w[1]) + (psi_mob * w[2])

            new_f = QgsFeature(out_fields)
            new_f.setGeometry(f.geometry())
            new_f.setAttributes(f.attributes() + [psi_edu, psi_heal, psi_mob, ip_dyn])
            sink.addFeature(new_f)
            feedback.setProgress(80 + int((i / len(grid_feats)) * 20))

        conn.close()
        return {self.OUTPUT: dest_id}
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

class ProximityEngineAttractiveness(QgsProcessingAlgorithm):
    INPUT_GRID = 'INPUT_GRID'
    LAYER_E_L = 'LAYER_E_L'; LAYER_E_M = 'LAYER_E_M'
    LAYER_H_L = 'LAYER_H_L'; LAYER_H_M = 'LAYER_H_M'
    LAYER_M_L = 'LAYER_M_L'; LAYER_M_M = 'LAYER_M_M'
    ATTR_FIELD = 'attr_weight' # Nome del campo peso nelle cover OSM
    DB_CONN = 'DB_CONN'
    EDGES_TABLE = 'EDGES_TABLE'
    GEOM_COL = 'GEOM_COL'
    TOLERANCE = 'TOLERANCE'
    OUTPUT = 'OUTPUT'

    def name(self): return 'proximity_3pillars_attractiveness'
    def displayName(self): return 'SNAI: Prossimità con Attrattività Servizi (Fuzzy)'
    def group(self): return 'Analisi Territoriale'
    def groupId(self): return 'prossimita'
    def createInstance(self): return ProximityEngineAttractiveness()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_GRID, 'Griglia Esagonale (w_edu, w_heal, w_mob)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_H_L, 'Salute Local (Field: attr_weight)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_H_M, 'Salute Macro (Field: attr_weight)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_E_L, 'Edu Local (Field: attr_weight)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_E_M, 'Edu Macro (Field: attr_weight)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_M_L, 'Mob Local (Field: attr_weight)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_M_M, 'Mob Macro (Field: attr_weight)'))
        
        self.addParameter(QgsProcessingParameterNumber(self.TOLERANCE, 'Snapping Tolerance (m)', defaultValue=1000))
        self.addParameter(QgsProcessingParameterString(self.DB_CONN, 'Conn. PostgreSQL', defaultValue="host=localhost dbname=osm user=postgres password=pwd"))
        self.addParameter(QgsProcessingParameterString(self.EDGES_TABLE, 'Tabella Archi (ways)', defaultValue="ways"))
        self.addParameter(QgsProcessingParameterString(self.GEOM_COL, 'Colonna Geom Vertici', defaultValue="the_geom"))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Risultato Prossimità Pesata'))

    def processAlgorithm(self, parameters, context, feedback):
        grid_source = self.parameterAsSource(parameters, self.INPUT_GRID, context)
        conn = psycopg2.connect(self.parameterAsString(parameters, self.DB_CONN, context))
        cur = conn.cursor()
        edges = self.parameterAsString(parameters, self.EDGES_TABLE, context)
        g_col = self.parameterAsString(parameters, self.GEOM_COL, context)
        tol = self.parameterAsDouble(parameters, self.TOLERANCE, context)

        # --- 1. SNAPPING CON RECUPERO ATTR_WEIGHT ---
        def snap_with_attr(layer_param, label):
            source = self.parameterAsSource(parameters, layer_param, context)
            if not source: return {}, {}
            node_map = {} # fid -> node_id
            attr_map = {} # node_id -> attr_weight
            
            for f in source.getFeatures():
                wkt = f.geometry().asWkt()
                a_weight = f[self.ATTR_FIELD] if self.ATTR_FIELD in f.fields().names() else 1.0
                
                sql = f"""
                    SELECT v.id FROM {edges}_vertices_pgr v
                    JOIN {edges} e ON (v.id = e.source OR v.id = e.target)
                    WHERE ST_Intersects(e.the_geom, ST_Transform(ST_Boundary(ST_GeomFromText('{wkt}', 32633)), 4326))
                    ORDER BY ST_Transform(v.{g_col}, 32633) <-> ST_GeomFromText('{wkt}', 32633) LIMIT 1
                """
                cur.execute(sql)
                res = cur.fetchone()
                if res:
                    nid = res[0]
                    node_map[f.id()] = nid
                    # Associe il peso al nodo (se più POI snappano sullo stesso nodo, prendiamo il max)
                    attr_map[nid] = max(attr_map.get(nid, 0), a_weight)
            return node_map, attr_map

        feedback.pushInfo("Snapping griglia e POI con pesi attrattività...")
        _, grid_map = snap_with_attr(self.INPUT_GRID, "Griglia") # Solo mapping per grid
        grid_nodes = list(set(grid_map.values()))
        
        poi_layers = {'el': self.LAYER_E_L, 'em': self.LAYER_E_M, 'hl': self.LAYER_H_L, 'hm': self.LAYER_H_M, 'ml': self.LAYER_M_L, 'mm': self.LAYER_M_M}
        poi_nodes = {}
        poi_attr = {}
        for k, lp in poi_layers.items():
            _, attr = snap_with_attr(lp, k)
            poi_nodes[k] = list(attr.keys())
            poi_attr[k] = attr

        # --- 2. ROUTING MATRICIALE (Dijkstra) ---
        feedback.pushInfo("Calcolo Routing...")
        costs = {} # {cat: {start_node: (min_cost, attr_weight_of_target)}}
        cfgs = {'el': 'cost_walk', 'em': 'cost_car', 'hl': 'cost_walk', 'hm': 'cost_car', 'ml': 'cost_walk', 'mm': 'cost_car'}
        
        for k, c_col in cfgs.items():
            if not poi_nodes[k]: continue
            # pgr_dijkstra restituisce il costo e il nodo finale (end_vid)
            # Dobbiamo sapere a QUALE servizio siamo arrivati per prenderne l'attr_weight
            sql = f"""
                SELECT start_vid, end_vid, agg_cost 
                FROM pgr_dijkstraCost(
                    'SELECT id, source, target, {c_col} as cost FROM {edges}',
                    ARRAY{grid_nodes}, ARRAY{poi_nodes[k]}, directed := false
                )
            """
            cur.execute(sql)
            costs[k] = {r[0]: (r[2], poi_attr[k].get(r[1], 1.0)) for r in cur.fetchall()}

        # --- 3. LOGICA FUZZY PESATA E OUTPUT ---
        def fuzzy_decay(sec, t0, stiffness, attr):
            if sec is None: return 0.0
            t = sec / 60.0
            score = 1 / (1 + math.exp(stiffness * (t - t0)))
            return score * attr # Moltiplica per l'attrattività del servizio più vicino

        # Creazione layer di output...
        # [Logica di aggregazione pilastri e pesi demografici identica allo script precedente]
        # psi_heal = (sc['hl'] * 0.4) + (sc['hm'] * 0.6) ... etc.
        
        # [Chiusura connessione e return]
        return {self.OUTPUT: "Completato"}

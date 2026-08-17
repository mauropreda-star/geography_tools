from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, 
                       QgsProcessingParameterFeatureSource, 
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterString,
                       QgsProcessingParameterNumber,
                       QgsFeature, QgsField, QgsProcessingException,
                       QgsSpatialIndex)
import psycopg2
import numpy as np
import math

class ProximityGenerator9VarsProgress(QgsProcessingAlgorithm):
    INPUT_GRID = 'INPUT_GRID'
    LAYER_EL = 'LAYER_EL'; LAYER_EM = 'LAYER_EM'
    LAYER_HL = 'LAYER_HL'; LAYER_HM = 'LAYER_HM'
    LAYER_ML = 'LAYER_ML'; LAYER_MM = 'LAYER_MM'
    DB_CONN = 'DB_CONN'
    EDGES_TABLE = 'EDGES_TABLE'
    GEOM_COL = 'GEOM_COL'
    TOLERANCE = 'TOLERANCE'
    OUTPUT = 'OUTPUT'

    def name(self): return 'proximity_generator_9vars_progress'
    def displayName(self): return '1. PROSSIMITÀ: Generatore Dataset RNA (9 vars + Progress)'
    def group(self): return 'Analisi Prossimità'
    def groupId(self): return 'prossimita'
    def createInstance(self): return ProximityGenerator9VarsProgress()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_GRID, 'Griglia Esagonale (32633) con w_edu, w_heal, w_mob'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_EL, 'Istruzione Local (Poly - 15m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_EM, 'Istruzione Macro (Auto - 60m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_HL, 'Salute Local (Poly - 15m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_HM, 'Salute Macro (Auto - 60m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_ML, 'Mobilità Local (Poly - 15m)'))
        self.addParameter(QgsProcessingParameterFeatureSource(self.LAYER_MM, 'Mobilità Macro (Auto - 60m)'))
        
        self.addParameter(QgsProcessingParameterNumber(self.TOLERANCE, 'Snapping Tolerance (m)', defaultValue=300))
        self.addParameter(QgsProcessingParameterString(self.DB_CONN, 'Conn. PostgreSQL (host=localhost dbname=... user=... password=...)'))
        self.addParameter(QgsProcessingParameterString(self.EDGES_TABLE, 'Tabella Archi pgRouting (4326)', defaultValue="ways"))
        self.addParameter(QgsProcessingParameterString(self.GEOM_COL, 'Colonna Geom Vertici DB', defaultValue="the_geom"))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 'Dataset Standardizzato RNA'))

    def processAlgorithm(self, parameters, context, feedback):
        grid_source = self.parameterAsSource(parameters, self.INPUT_GRID, context)
        conn = psycopg2.connect(self.parameterAsString(parameters, self.DB_CONN, context))
        cur = conn.cursor()
        edges = self.parameterAsString(parameters, self.EDGES_TABLE, context)
        g_col = self.parameterAsString(parameters, self.GEOM_COL, context)
        tol = self.parameterAsDouble(parameters, self.TOLERANCE, context)

        # --- FASE 1: SNAPPING GRIGLIA (0% - 15%) ---
        feedback.pushInfo("Fase 1/6: Snapping centroidi griglia alla rete...")
        grid_node_map = {}
        grid_features = {f.id(): f for f in grid_source.getFeatures()}
        total_grid = len(grid_features)
        
        for i, (fid, f) in enumerate(grid_features.items()):
            if feedback.isCanceled(): return {}
            c = f.geometry().centroid().asPoint()
            sql = f"SELECT id FROM {edges}_vertices_pgr v WHERE ST_DWithin(ST_Transform(v.{g_col}, 32633), ST_SetSRID(ST_Point({c.x()},{c.y()}), 32633), {tol}) ORDER BY v.{g_col} <-> ST_Transform(ST_SetSRID(ST_Point({c.x()},{c.y()}), 32633), 4326) LIMIT 1"
            cur.execute(sql)
            res = cur.fetchone()
            grid_node_map[fid] = res[0] if res else None
            if i % 100 == 0: feedback.setProgress(int((i / total_grid) * 15))

        # --- FASE 2: SNAPPING POI (15% - 30%) ---
        feedback.pushInfo("Fase 2/6: Snapping poligoni servizi alla rete...")
        poi_layers_map = {'el': self.LAYER_EL, 'em': self.LAYER_EM, 'hl': self.LAYER_HL, 'hm': self.LAYER_HM, 'ml': self.LAYER_ML, 'mm': self.LAYER_MM}
        poi_nodes = {}
        poi_sources = {}
        for idx, (k, lp) in enumerate(poi_layers_map.items()):
            if feedback.isCanceled(): return {}
            source = self.parameterAsSource(parameters, lp, context)
            poi_sources[k] = source
            if not source: poi_nodes[k] = []; continue
            
            wkt_list = [f"ST_GeomFromText('{f.geometry().asWkt()}', 32633)" for f in source.getFeatures()]
            if not wkt_list: poi_nodes[k] = []; continue

            sql = f"SELECT DISTINCT v.id FROM {edges}_vertices_pgr v WHERE EXISTS (SELECT 1 FROM (SELECT unnest(ARRAY[{','.join(wkt_list)}]) as poly) as sub WHERE ST_Intersects(ST_Transform(v.{g_col}, 32633), sub.poly) OR ST_DWithin(ST_Transform(v.{g_col}, 32633), sub.poly, 50))"
            cur.execute(sql)
            poi_nodes[k] = [row[0] for row in cur.fetchall()]
            feedback.setProgress(15 + int(((idx + 1) / 6) * 15))

        # --- FASE 3: ANALISI INCLUSIONI (30% - 45%) ---
        feedback.pushInfo("Fase 3/6: Verifica inclusioni centroidi (Costo 0.1)...")
        is_inside = {k: set() for k in poi_layers_map.keys()}
        for idx, (k, source) in enumerate(poi_sources.items()):
            if not source: continue
            spatial_index = QgsSpatialIndex(source.getFeatures())
            poi_feats = {f.id(): f for f in source.getFeatures()}
            for fid, f_grid in grid_features.items():
                if feedback.isCanceled(): return {}
                centroid = f_grid.geometry().centroid()
                candidates = spatial_index.intersects(centroid.boundingBox())
                for c_id in candidates:
                    if poi_feats[c_id].geometry().contains(centroid):
                        is_inside[k].add(fid)
                        break
            feedback.setProgress(30 + int(((idx + 1) / 6) * 15))

        # --- FASE 4: ROUTING BATCH (45% - 85%) ---
        feedback.pushInfo("Fase 4/6: Calcolo Routing Matriciale pgRouting (Batching)...")
        unique_grid_nodes = list(set([n for n in grid_node_map.values() if n]))
        costs = {k: {} for k in poi_layers_map.keys()}
        cfgs = {'el': 'cost_walk', 'em': 'cost_car', 'hl': 'cost_walk', 'hm': 'cost_car', 'ml': 'cost_walk', 'mm': 'cost_car'}
        
        BATCH_SIZE = 100
        for cat_idx, (k, c_col) in enumerate(cfgs.items()):
            if not poi_nodes[k] or not unique_grid_nodes: continue
            feedback.pushInfo(f"  > Elaborazione pilastro {k}...")
            for b_idx in range(0, len(unique_grid_nodes), BATCH_SIZE):
                if feedback.isCanceled(): return {}
                batch = unique_grid_nodes[b_idx : b_idx + BATCH_SIZE]
                sql = f"SELECT start_vid, min(agg_cost) FROM pgr_dijkstraCost('SELECT id, source, target, {c_col} as cost FROM {edges}', ARRAY{batch}, ARRAY{poi_nodes[k]}, directed := false) GROUP BY start_vid"
                cur.execute(sql)
                costs[k].update(dict(cur.fetchall()))
                
                # Progresso basato sui batch e sulla categoria
                prog = 45 + (cat_idx / 6 * 40) + (b_idx / len(unique_grid_nodes) * (40/6))
                feedback.setProgress(int(prog))

        # --- FASE 5: FINALIZZAZIONE E PESI (85% - 100%) ---
        feedback.pushInfo("Fase 5/6: Scrittura layer finale a 9 variabili...")
        out_fields = grid_source.fields()
        # Rimuoviamo campi con lo stesso nome se già esistenti per evitare errori
        ai_vars = ["psi_el", "psi_em", "psi_hl", "psi_hm", "psi_ml", "psi_mm", "w_edu_n", "w_heal_n", "w_mob_n", "ip_dyn"]
        for v in ai_vars:
            idx = out_fields.indexFromName(v)
            if idx != -1: out_fields.remove(idx)
            out_fields.append(QgsField(v, QVariant.Double))
        
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, grid_source.wkbType(), grid_source.sourceCrs())

        def fuzzy_decay(sec, t0, stiffness):
            if sec is None: return 0.0
            t = sec / 60.0
            return 1 / (1 + math.exp(stiffness * (t - t0)))

        for i, (fid, f) in enumerate(grid_features.items()):
            if feedback.isCanceled(): break
            nid = grid_node_map.get(fid)
            sc = {}
            for k in cfgs.keys():
                if fid in is_inside[k]: sc[k] = 1.0
                else:
                    c_sec = costs[k].get(nid) if nid else None
                    t0 = 15 if k.endswith('l') else 60
                    stiff = 0.3 if k.endswith('l') else 0.1
                    sc[k] = fuzzy_decay(c_sec, t0, stiff)

            # Normalizzazione Pesi ISTAT dalla griglia (Fix IP_dyn)
            try:
                w_e = float(f['w_edu'] or 0.333)
                w_h = float(f['w_heal'] or 0.333)
                w_m = float(f['w_mob'] or 0.334)
            except:
                w_e, w_h, w_m = 0.333, 0.333, 0.334

            w_sum = w_e + w_h + w_m
            w_en, w_hn, w_mn = (w_e/w_sum, w_h/w_sum, w_m/w_sum) if w_sum > 0 else (0.333, 0.333, 0.334)

            # Calcolo target ip_dyn (Logica Fuzzy Aggregata del Capitolo 4)
            p_edu = (sc['el']*0.6 + sc['em']*0.4)
            p_heal = (sc['hl']*0.4 + sc['hm']*0.6)
            p_mob = (sc['ml']*0.5 + sc['mm']*0.5)
            ip_dyn = (p_edu * w_en) + (p_heal * w_hn) + (p_mob * w_mn)

            new_f = QgsFeature(out_fields)
            new_f.setGeometry(f.geometry())
            new_f.setAttributes(f.attributes() + [sc['el'], sc['em'], sc['hl'], sc['hm'], sc['ml'], sc['mm'], w_en, w_hn, w_mn, ip_dyn])
            sink.addFeature(new_f)
            if i % 100 == 0: feedback.setProgress(85 + int((i/total_grid)*15))

        # --- FASE 6: CHIUSURA ---
        conn.close()
        feedback.pushInfo("Fase 6/6: Elaborazione conclusa con successo.")
        return {self.OUTPUT: dest_id}
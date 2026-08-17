-- 1. Aggiunta delle colonne dei pesi alla tabella grid_hex se non esistono
ALTER TABLE  statistiche.hex_scenario_roma_subiaco  ADD COLUMN IF NOT EXISTS w_heal double precision;
ALTER TABLE  statistiche.hex_scenario_roma_subiaco  ADD COLUMN IF NOT EXISTS w_edu double precision;
ALTER TABLE  statistiche.hex_scenario_roma_subiaco  ADD COLUMN IF NOT EXISTS w_mob double precision;
-- 2. Calcolo dei pesi tramite CTE (Common Table Expressions)
WITH spatial_interpolation AS (
    -- Fase A: Trasferimento proporzionale dei dati dalle sezioni agli esagoni
    SELECT 
        h.id,
        SUM(s.p1 * ST_Area(ST_Intersection(h.geom, s.geom)) / ST_Area(s.geom)) as pop_tot,
        SUM((s.p27 + s.p28 + s.p29) * ST_Area(ST_Intersection(h.geom, s.geom)) / ST_Area(s.geom)) as pop_anziani,
        SUM((s.p14 + s.p15 + s.p17 + s.p18 + s.p19) * ST_Area(ST_Intersection(h.geom, s.geom)) / ST_Area(s.geom)) as pop_istruzione
    FROM statistiche.hex_scenario_roma_subiaco h
    JOIN statistiche.sez_cens_12_2021_j_data s ON ST_Intersects(h.geom, s.geom)
    GROUP BY h.id
),
normalized_indices AS (
    -- Fase B: Calcolo degli indici di domanda (0-1)
    -- Gestiamo la divisione per zero per le aree non abitate
    SELECT 
        id,
        CASE WHEN pop_tot > 0 THEN pop_anziani / pop_tot ELSE 0 END as i_heal,
        CASE WHEN pop_tot > 0 THEN pop_istruzione / pop_tot ELSE 0 END as i_edu,
        -- Per la mobilità usiamo la densità di popolazione come proxy del bisogno di TPL
        pop_tot / 106062 as i_mob -- valore mq è l'area di un esagono da 500m
    FROM spatial_interpolation
),
propensities AS (
    -- Fase C: Calcolo delle propensioni grezze (k) con moltiplicatori beta
    -- Baseline paritaria 0.333
    SELECT 
        id,
        (0.333 + (i_heal * 0.5)) as k_heal,
        (0.333 + (i_edu * 0.4)) as k_edu,
        (0.333 + (i_mob * 0.3)) as k_mob
    FROM normalized_indices
)
-- 3. Update finale della griglia con normalizzazione unitaria (Somma = 1)
UPDATE statistiche.hex_scenario_roma_subiaco h
SET 
    w_heal = p.k_heal / (p.k_heal + p.k_edu + p.k_mob),
    w_edu = p.k_edu / (p.k_heal + p.k_edu + p.k_mob),
    w_mob = p.k_mob / (p.k_heal + p.k_edu + p.k_mob)
FROM propensities p
WHERE h.id = p.id;
-- 4. Pulizia per esagoni non abitati (opzionale, assegna peso paritario 1/3)
UPDATE statistiche.hex_scenario_roma_subiaco
SET w_heal = 0.333, w_edu = 0.333, w_mob = 0.334 
WHERE w_heal IS NULL;


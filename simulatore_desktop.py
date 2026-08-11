import os
import sys
import threading
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import joblib
from tensorflow.keras.models import load_model
from shapely.geometry import box
from scipy.spatial import KDTree

# --- CONFIGURAZIONE AMBIENTE ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    sys.exit("Errore: Tkinter non trovato.")

try:
    import mapclassify
    MAPCLASSIFY_AVAILABLE = True
except ImportError:
    MAPCLASSIFY_AVAILABLE = False

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except Exception:
    CONTEXTILY_AVAILABLE = False

class ProximityNeuralFieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PROSSIMITÀ 2026 - Neural Field Simulator v3.5 ")
        self.root.geometry("1600x920")
        self.root.resizable(False, False)
        
        # Stato Dati
        self.gdf = None
        self.model = None
        self.scaler = None
        self.tree = None
        self.selected_indices = []
        self.is_calculating = False
        self.selector_active = False
        
        self.features_9 = ["psi_el", "psi_em", "psi_hl", "psi_hm", "psi_ml", "psi_mm", "w_edu", "w_heal", "w_mob"]
        self.cbar = None
        
        self.setup_ui()

    def setup_ui(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # --- SIDEBAR SCROLLABILE ---
        self.sidebar_container = tk.Frame(self.paned, width=420, bg="#f8f9fa")
        self.paned.add(self.sidebar_container)

        self.canvas_sidebar = tk.Canvas(self.sidebar_container, bg="#f8f9fa", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.sidebar_container, orient="vertical", command=self.canvas_sidebar.yview)
        self.scrollable_sidebar = tk.Frame(self.canvas_sidebar, bg="#f8f9fa")

        self.scrollable_sidebar.bind("<Configure>", lambda e: self.canvas_sidebar.configure(scrollregion=self.canvas_sidebar.bbox("all")))
        self.canvas_sidebar.create_window((0, 0), window=self.scrollable_sidebar, anchor="nw")
        self.canvas_sidebar.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_sidebar.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        s = self.scrollable_sidebar
        ttk.Label(s, text="🎮 AI FIELD CONTROL", font=('Helvetica', 16, 'bold')).pack(pady=10, padx=20)
        
        # 1. Dati & Modello
        f_io = tk.LabelFrame(s, text=" 1. Caricamento Asset ", bg="#f8f9fa", padx=10, pady=5)
        f_io.pack(fill=tk.X, padx=15, pady=5)
        ttk.Button(f_io, text="📂 Carica Griglia GeoJSON (sr 4326)", command=self.load_data).pack(fill=tk.X, pady=2)
        ttk.Button(f_io, text="🧠 Carica Modello RNA", command=self.load_ai).pack(fill=tk.X, pady=2)

        # 2. Selezione
        f_sel = tk.LabelFrame(s, text=" 2. Editing Territoriale ", bg="#f8f9fa", padx=10, pady=5)
        f_sel.pack(fill=tk.X, padx=15, pady=5)
        self.sel_label = ttk.Label(f_sel, text="Celle selezionate: 0", font=('Arial', 10, 'bold'), foreground="blue")
        self.sel_label.pack(pady=2)
        self.btn_sel_mode = tk.Button(f_sel, text="ABILITA Selezione Rettangolo", bg="#ffc107", command=self.toggle_selector)
        self.btn_sel_mode.pack(fill=tk.X, pady=2)
        ttk.Button(f_sel, text="✏️ Edita Selezione", command=self.open_bulk_edit).pack(fill=tk.X, pady=2)
        ttk.Button(f_sel, text="🗑️ Deseleziona", command=self.clear_selection).pack(fill=tk.X, pady=2)

        # 3. Progress & Ricalcolo
        f_run = tk.LabelFrame(s, text=" 3. Sincronizzazione AI ", bg="#f8f9fa", padx=10, pady=5)
        f_run.pack(fill=tk.X, padx=15, pady=5)
        self.prog_bar = ttk.Progressbar(f_run, mode='determinate')
        self.prog_bar.pack(fill=tk.X, pady=5)
        self.status_lbl = ttk.Label(f_run, text="Pronto.", font=('Arial', 9))
        self.status_lbl.pack()
        self.cmap_combo = ttk.Combobox(f_run, values=["Greys_r", "RdYlGn", "viridis", "plasma"], state="readonly")
        self.cmap_combo.set("Greys_r")
        self.cmap_combo.pack(fill=tk.X, pady=5)
        self.btn_run = tk.Button(f_run, text="🚀 AVVIA SINCRONIZZAZIONE", bg="#d9534f", fg="white", font=('Helvetica', 12, 'bold'), command=self.start_async_calc)
        self.btn_run.pack(fill=tk.X, pady=5)

        # 4. Scheda Dati Hover
        f_hover = tk.LabelFrame(s, text=" 4. Scheda Dati Cella ", bg="#f8f9fa", padx=10, pady=5)
        f_hover.pack(fill=tk.X, padx=15, pady=5)
        self.hover_info = tk.Text(f_hover, height=18, font=('Consolas', 10), bg="#212529", fg="#00ff00", padx=10, pady=10)
        self.hover_info.pack(fill=tk.X)

        # --- MAPPA ---
        self.map_container = tk.Frame(self.paned, bg="white")
        self.paned.add(self.map_container)
        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.85, 0.9])
        self.ax_cb = self.fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_container)

        # Connessione Eventi (Hover sempre attivo)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_map_click)

    # --- LOGICA HOVER MIGLIORATA ---
    def on_mouse_move(self, event):
        """Gestisce la visualizzazione dei dati in tempo reale, anche dopo lo zoom"""
        if self.gdf is None or self.tree is None or self.is_calculating:
            return
        
        # Verifica se il mouse è nell'asse della mappa (ignora la colorbar)
        if event.inaxes != self.ax or self.selector_active:
            return

        # Riceve le coordinate del mouse (in metri EPSG:3857)
        try:
            dist, idx = self.tree.query([event.xdata, event.ydata])
            
            # Soglia dinamica: usiamo una distanza fissa in metri 
            # (500m è la dimensione media dell'esagono)
            if dist < 600: 
                r = self.gdf.iloc[idx]
                self.hover_info.config(state='normal')
                self.hover_info.delete('1.0', tk.END)
                
                info = f"ID_HEX: {r['id_hex']}\n"
                info += "="*25 + "\n"
                for v in self.features_9:
                    info += f"{v.upper():<12}: {r.get(v, 0):.3f}\n"
                
                if 'ip_dyn' in r:
                    info += "-"*25 + "\n"
                    info += f"IP DINAMICO: {r['ip_dyn']:.4f}\n"
                
                self.hover_info.insert(tk.END, info)
                self.hover_info.config(state='disabled')
            else:
                # Se siamo fuori dalla griglia, puliamo se necessario
                pass
        except Exception:
            pass

    def refresh_map(self):
        """Ridisegna la mappa mantenendo la stabilità del layout"""
        # Salva i limiti attuali prima di pulire (per mantenere lo zoom)
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        self.ax.clear(); self.ax_cb.clear()
        self.ax.set_axis_off(); self.ax_cb.set_axis_off()

        if self.gdf is not None:
            col = 'ip_dyn' if 'ip_dyn' in self.gdf.columns else None
            cmap = self.cmap_combo.get()
            
            if col:
                self.gdf.plot(column=col, ax=self.ax, cmap=cmap, alpha=0.6, 
                              scheme='quantiles', k=5, edgecolor='black', linewidth=0.05)
                self.ax_cb.set_axis_on()
                norm = plt.Normalize(vmin=self.gdf[col].min(), vmax=self.gdf[col].max())
                self.fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=self.ax_cb)
            else:
                self.gdf.plot(ax=self.ax, facecolor='#e0e0e0', edgecolor='black', linewidth=0.1)

            if self.selected_indices:
                self.gdf.iloc[self.selected_indices].plot(ax=self.ax, facecolor='cyan', alpha=0.5, edgecolor='blue', linewidth=1)

            if CONTEXTILY_AVAILABLE:
                try: ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik)
                except: pass
        
        # Ripristina i limiti dello zoom se non è la prima volta che carichiamo
        if cur_xlim != (0.0, 1.0): # Matplotlib default init limits
            self.ax.set_xlim(cur_xlim)
            self.ax.set_ylim(cur_ylim)

        if self.selector_active:
            self.rect_selector = RectangleSelector(self.ax, self.on_rectangle_select, useblit=True, button=[1], props=dict(facecolor='cyan', alpha=0.3))
        
        self.canvas.draw()

    # --- LOGICA AI ---
    def start_async_calc(self):
        if self.gdf is None or self.model is None:
            messagebox.showwarning("Dati mancanti", "Carica GeoJSON e Modello RNA.")
            return
        self.is_calculating = True
        self.btn_run.config(state=tk.DISABLED)
        threading.Thread(target=self.run_neural_logic, daemon=True).start()

    def run_neural_logic(self):
        total = len(self.gdf)
        coords = np.array(list(zip(self.gdf.geometry.centroid.x, self.gdf.geometry.centroid.y)))
        data_matrix = self.gdf[self.features_9].values
        R_LOCAL, R_MACRO = 1500, 20000
        def fw(d, t0, stiff): return 1 / (1 + np.exp(stiff * (d/100 - t0)))
        all_27 = []
        for i in range(total):
            p = coords[i]
            idx_l = self.tree.query_ball_point(p, R_LOCAL)
            v_l = np.sum(data_matrix[idx_l].T * fw(np.linalg.norm(coords[idx_l]-p, axis=1), 15, 0.3), axis=1) / np.sum(fw(np.linalg.norm(coords[idx_l]-p, axis=1), 15, 0.3)) if idx_l else np.zeros(9)
            idx_m = self.tree.query_ball_point(p, R_MACRO)
            v_m = np.sum(data_matrix[idx_m].T * fw(np.linalg.norm(coords[idx_m]-p, axis=1), 60, 0.1), axis=1) / np.sum(fw(np.linalg.norm(coords[idx_m]-p, axis=1), 60, 0.1)) if idx_m else np.zeros(9)
            all_27.append(np.concatenate([data_matrix[i], v_l, v_m]))
            if i % 100 == 0: self.prog_bar['value'] = (i / total) * 100
        X_scaled = self.scaler.transform(np.array(all_27))
        self.gdf['ip_dyn'] = self.model.predict(X_scaled, verbose=0).flatten()
        self.root.after(0, self.finish_calc)

    def finish_calc(self):
        self.refresh_map(); self.is_calculating = False; self.btn_run.config(state=tk.NORMAL)
        self.prog_bar['value'] = 0; self.status_lbl.config(text="Pronto.")

    # --- IO & EDITING ---
    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("GeoData", "*.geojson"), ("GeoPackage", "*.gpkg")])
        if path:
            self.gdf = gpd.read_file(path)
            self.gdf['id_hex'] = self.gdf['id_hex'].astype(str)
            self.gdf = self.gdf.to_crs(epsg=3857)
            coords = np.array(list(zip(self.gdf.geometry.centroid.x, self.gdf.geometry.centroid.y)))
            self.tree = KDTree(coords)
            self.refresh_map()

    def load_ai(self):
        m_path = filedialog.askopenfilename(filetypes=[("Keras", "*.keras")])
        s_path = filedialog.askopenfilename(filetypes=[("Scaler", "*.gz")])
        if m_path and s_path:
            self.model = load_model(m_path, compile=False)
            self.scaler = joblib.load(s_path)
            messagebox.showinfo("AI", "Asset Caricati.")

    def toggle_selector(self):
        if not self.selector_active:
            self.selector_active = True
            self.btn_sel_mode.config(text="DISATTIVA SELEZIONE", bg="#dc3545", fg="white")
            self.rect_selector = RectangleSelector(self.ax, self.on_rectangle_select, useblit=True, button=[1], props=dict(facecolor='cyan', alpha=0.3))
        else:
            self.selector_active = False
            self.btn_sel_mode.config(text="ABILITA SELEZIONE", bg="#ffc107", fg="black")
            self.refresh_map()

    def on_rectangle_select(self, eclick, erelease):
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        matches = self.gdf[self.gdf.geometry.intersects(box(xmin, ymin, xmax, ymax))]
        self.selected_indices = matches.index.tolist()
        self.sel_label.config(text=f"Celle: {len(self.selected_indices)}")
        self.refresh_map()

    def clear_selection(self):
        self.selected_indices = []; self.sel_label.config(text="Celle: 0"); self.refresh_map()

    def open_bulk_edit(self):
        if not self.selected_indices: return
        pop = tk.Toplevel(self.root); pop.title("Edit 9 Vars"); pop.geometry("350x550"); pop.attributes("-topmost", True)
        entries = {}
        ref = self.gdf.iloc[self.selected_indices[0]]
        for f in self.features_9:
            tk.Label(pop, text=f.upper()).pack()
            e = ttk.Entry(pop); e.insert(0, str(round(ref[f], 3))); e.pack(); entries[f] = e
        def apply():
            for f, e in entries.items(): self.gdf.loc[self.selected_indices, f] = float(e.get().replace(',','.'))
            pop.destroy(); self.refresh_map()
        ttk.Button(pop, text="Applica", command=apply).pack(pady=20)

    def on_map_click(self, event):
        if self.toolbar.mode != "" or self.gdf is None or self.selector_active or event.xdata is None: return
        dist, idx = self.tree.query([event.xdata, event.ydata])
        if dist < 600:
            if idx in self.selected_indices: self.selected_indices.remove(idx)
            else: self.selected_indices.append(idx)
            self.sel_label.config(text=f"Celle: {len(self.selected_indices)}"); self.refresh_map()

if __name__ == "__main__":
    root = tk.Tk(); app = ProximityNeuralFieldApp(root); root.mainloop()
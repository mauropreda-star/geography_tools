import os
import sys
import threading
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
import joblib
from tensorflow.keras.models import load_model
from shapely.geometry import Point, LineString
from scipy.spatial import KDTree
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- CONFIGURAZIONE AMBIENTE ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False

class ProximityNeuroDesigner:
    def __init__(self, root):
        self.root = root
        self.root.title("PROSSIMITÀ 2026 - Simulazione AI  v4.0")
        self.root.geometry("1650x950")
        
        # Stato
        self.gdf = None
        self.model = None
        self.scaler = None
        self.tree = None
        self.new_services = [] 
        self.is_calculating = False
        self.tool_mode = "Navigate" 
        self.profile_pts = []
        self.dynamic_circle = None 
        self.cbar = None

        # Variabili UI
        self.add_mode_flag = tk.BooleanVar(value=False)
        self.attr_weight_val = tk.DoubleVar(value=1.0)
        self.lens_radius = tk.DoubleVar(value=1500.0)
        
        self.marker_config = {
            "Salute":    {"marker": "P", "color": "#d9534f"},
            "Istruzione": {"marker": "*", "color": "#0275d8"},
            "Mobilità":  {"marker": "^", "color": "#5cb85c"}
        }
        
        self.base_features = ["psi_hl", "psi_hm", "psi_el", "psi_em", "psi_ml", "psi_mm", "w_edu", "w_heal", "w_mob"]
        self.all_items = self.base_features + ["ip_dyn"]
        
        # Mappatura Scale Cromatiche
        self.cmap_dict = {
            "Nero (0) -> Bianco (1)": "Greys_r",
            "Blu (0) -> Rosso (1)": "RdBu_r",
            "Verde (0) -> Rosso (1)": "RdYlGn_r"
        }
        
        self.setup_ui()

    def setup_ui(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.paned, width=420, bg="#f8f9fa")
        self.paned.add(self.sidebar)
        
        # 1. Asset
        f_io = tk.LabelFrame(self.sidebar, text=" 1. Asset ", bg="#f8f9fa", padx=10, pady=5)
        f_io.pack(fill=tk.X, padx=15, pady=5)
        self.btn_load_grid = ttk.Button(f_io, text="📂 Carica Griglia (.geojson)", command=self.load_data)
        self.btn_load_grid.pack(fill=tk.X, pady=2)
        self.btn_load_ai = ttk.Button(f_io, text="🧠 Carica Modello (.keras)", command=self.load_ai)
        self.btn_load_ai.pack(fill=tk.X, pady=2)
        self.btn_load_scaler = ttk.Button(f_io, text="⚖️ Carica Scaler (.gz)", command=self.load_scaler)
        self.btn_load_scaler.pack(fill=tk.X, pady=2)

        # 2. Diagnostica
        f_tools = tk.LabelFrame(self.sidebar, text=" 2. Diagnostica ", bg="#f8f9fa", padx=10, pady=5)
        f_tools.pack(fill=tk.X, padx=15, pady=5)
        self.btn_profile = tk.Button(f_tools, text="📈 Profilo Altimetrico", command=lambda: self.set_tool("Profile"), bg="#28a745", fg="white", font=('Arial', 9, 'bold'))
        self.btn_profile.pack(fill=tk.X, pady=2)
        self.btn_lens = tk.Button(f_tools, text="🔍 Lente territoriale", command=lambda: self.set_tool("Lens"), bg="#28a745", fg="white", font=('Arial', 9, 'bold'))
        self.btn_lens.pack(fill=tk.X, pady=2)
        f_rad = tk.Frame(f_tools, bg="#f8f9fa"); f_rad.pack(pady=2)
        ttk.Label(f_rad, text="Raggio (m):").pack(side=tk.LEFT); ttk.Entry(f_rad, textvariable=self.lens_radius, width=10).pack(side=tk.LEFT, padx=5)

        # 3. Rammendo
        f_plan = tk.LabelFrame(self.sidebar, text=" 3. Modulazione ", bg="#f8f9fa", padx=10, pady=5)
        f_plan.pack(fill=tk.X, padx=15, pady=5)
        tk.Checkbutton(f_plan, text="MODALITÀ INSERIMENTO", variable=self.add_mode_flag, font=('Arial', 9, 'bold'), fg="green").pack(anchor='w')
        self.pillar_cb = ttk.Combobox(f_plan, values=["Salute", "Istruzione", "Mobilità"], state="readonly"); self.pillar_cb.set("Salute"); self.pillar_cb.pack(fill=tk.X, pady=2)
        self.scale_cb = ttk.Combobox(f_plan, values=["Local (15 min)", "Macro (60 min)"], state="readonly"); self.scale_cb.set("Local (15 min)"); self.scale_cb.pack(fill=tk.X, pady=2)
        tk.Scale(f_plan, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.attr_weight_val, label="Attrattività").pack(fill=tk.X)

        # 4. Visualizzazione (Upgrade v4.6)
        f_viz = tk.LabelFrame(self.sidebar, text=" 4. Visualizzazione ", bg="#f8f9fa", padx=10, pady=5)
        f_viz.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(f_viz, text="Indicatore:").pack(anchor='w')
        self.map_col_cb = ttk.Combobox(f_viz, values=self.all_items, state="readonly"); self.map_col_cb.set("ip_dyn"); self.map_col_cb.pack(fill=tk.X, pady=2)
        self.map_col_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_map())
        
        ttk.Label(f_viz, text="Scala Cromatica:").pack(anchor='w')
        self.cmap_cb = ttk.Combobox(f_viz, values=list(self.cmap_dict.keys()), state="readonly")
        self.cmap_cb.set("Nero (0) -> Bianco (1)"); self.cmap_cb.pack(fill=tk.X, pady=2)
        self.cmap_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_map())

        # Monitoraggio
        self.hover_info = tk.Text(self.sidebar, height=12, font=('Consolas', 10), bg="#212529", fg="#00ff00", padx=10, pady=10)
        self.hover_info.pack(fill=tk.X, padx=15, pady=5)
        self.prog_bar = ttk.Progressbar(self.sidebar, mode='determinate'); self.prog_bar.pack(fill=tk.X, padx=20, pady=5)
        self.status_lbl = ttk.Label(self.sidebar, text="Pronto.", font=('Arial', 9), background="#f8f9fa"); self.status_lbl.pack()
        self.btn_run = tk.Button(self.sidebar, text="🚀 SINCRONIZZA AI", bg="#d9534f", fg="white", font=('Helvetica', 12, 'bold'), command=self.start_sync)
        self.btn_run.pack(fill=tk.X, padx=20, pady=10)

        # Area Mappa
        self.map_container = tk.Frame(self.paned, bg="white")
        self.paned.add(self.map_container)
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_axes([0.05, 0.05, 0.8, 0.9])
        self.ax_cb = self.fig.add_axes([0.88, 0.15, 0.02, 0.7])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_container)
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def set_tool(self, mode):
        if self.tool_mode == mode: self.tool_mode = "Navigate"
        else: self.tool_mode = mode; self.profile_pts = []
        self.btn_profile.config(bg="#ff9999" if self.tool_mode=="Profile" else "#28a745")
        self.btn_lens.config(bg="#ff9999" if self.tool_mode=="Lens" else "#28a745")
        self.refresh_map()

    def refresh_map(self):
        self.ax.clear(); self.ax_cb.clear(); self.ax.set_axis_off(); self.ax_cb.set_axis_off()
        if self.gdf is not None:
            col = self.map_col_cb.get()
            cmap_name = self.cmap_dict[self.cmap_cb.get()]
            if col in self.gdf.columns:
                # RENDERING CON 15 QUANTILI
                if self.gdf[col].nunique() > 15:
                    self.gdf.plot(column=col, ax=self.ax, cmap=cmap_name, scheme='quantiles', k=15, alpha=0.55, edgecolor='black', linewidth=0.01)
                    self.ax_cb.set_axis_on()
                    norm = plt.Normalize(vmin=self.gdf[col].min(), vmax=self.gdf[col].max())
                    self.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name), cax=self.ax_cb)
                else:
                    self.gdf.plot(column=col, ax=self.ax, cmap=cmap_name, alpha=0.55, edgecolor='black', linewidth=0.01)
            
            for s in self.new_services:
                cfg = self.marker_config[s['pillar']]
                self.ax.scatter(s['x'], s['y'], marker=cfg['marker'], color=cfg['color'], s=150, edgecolors='black', zorder=20)
            
            if CONTEXTILY_AVAILABLE:
                try: ctx.add_basemap(self.ax, source=ctx.providers.CartoDB.Voyager)
                except: pass
        self.canvas.draw()

    # --- LOGICA OPERATIVA ---
    def on_map_click(self, event):
        if event.xdata is None or self.gdf is None: return
        if self.add_mode_flag.get():
            self.new_services.append({'x': event.xdata, 'y': event.ydata, 'pillar': self.pillar_cb.get(), 'scale': self.scale_cb.get(), 'weight': self.attr_weight_val.get()})
            self.refresh_map(); return
        if self.tool_mode == "Profile":
            self.profile_pts.append((event.xdata, event.ydata))
            if len(self.profile_pts) == 2: self.show_profile_window(); self.profile_pts = []
        elif self.tool_mode == "Lens": self.show_lens_window(event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if self.gdf is None or event.inaxes != self.ax or self.is_calculating: return
        if self.tool_mode == "Lens":
            if self.dynamic_circle: 
                try: self.dynamic_circle.remove()
                except: pass
            self.dynamic_circle = Circle((event.xdata, event.ydata), self.lens_radius.get(), color='blue', alpha=0.1, linestyle='--')
            self.ax.add_patch(self.dynamic_circle); self.canvas.draw_idle()
        dist, idx = self.tree.query([event.xdata, event.ydata])
        if dist < 600:
            r = self.gdf.iloc[idx]
            self.hover_info.config(state='normal'); self.hover_info.delete('1.0', tk.END)
            for v in self.all_items: self.hover_info.insert(tk.END, f"{v.upper():<12}: {r.get(v,0):.3f}\n")
            self.hover_info.config(state='disabled')

    def start_sync(self):
        if self.gdf is None or self.model is None or self.scaler is None: return
        self.btn_run.config(state=tk.DISABLED); self.is_calculating = True
        threading.Thread(target=self.run_sync_logic, daemon=True).start()

    def run_sync_logic(self):
        coords = np.array(list(zip(self.gdf.geometry.centroid.x, self.gdf.geometry.centroid.y)))
        data_matrix = self.gdf[self.base_features].values
        for s in self.new_services:
            col = "psi_hl" if s['pillar']=="Salute" and "Local" in s['scale'] else "psi_hm" if s['pillar']=="Salute" else \
                  "psi_el" if s['pillar']=="Istruzione" and "Local" in s['scale'] else "psi_em" if s['pillar']=="Istruzione" else \
                  "psi_ml" if "Local" in s['scale'] else "psi_mm"
            t0 = 15 if "Local" in s['scale'] else 60
            dists = np.linalg.norm(coords - [s['x'], s['y']], axis=1)
            new_psi = (1 / (1 + np.exp(0.3 * (dists/100 - t0)))) * s['weight']
            self.gdf[col] = np.maximum(self.gdf[col], new_psi)
        
        self.gdf['psi_heal'] = (self.gdf['psi_hl']*0.4) + (self.gdf['psi_hm']*0.6)
        self.gdf['psi_edu'] = (self.gdf['psi_el']*0.6) + (self.gdf['psi_em']*0.4)
        self.gdf['psi_mob'] = (self.gdf['psi_ml']*0.5) + (self.gdf['psi_mm']*0.5)
        
        all_27 = []
        for i in range(len(self.gdf)):
            p_t = coords[i]
            def get_s(r, t0, st):
                idx = self.tree.query_ball_point(p_t, r)
                if not idx: return np.zeros(9)
                d = np.linalg.norm(coords[idx] - p_t, axis=1)
                w = 1 / (1 + np.exp(st * (d/100 - t0)))
                return np.sum(self.gdf.iloc[idx][self.base_features].values.T * w, axis=1) / np.sum(w) if np.sum(w)>0 else np.zeros(9)
            all_27.append(np.concatenate([data_matrix[i], get_s(1500, 15, 0.3), get_s(20000, 60, 0.1)]))
            if i % 100 == 0: self.prog_bar['value'] = (i/len(self.gdf))*100
        X_scaled = self.scaler.transform(np.array(all_27))
        self.gdf['ip_dyn'] = self.model.predict(X_scaled, verbose=0).flatten()
        self.root.after(0, self.refresh_map)
        self.root.after(0, lambda: [self.btn_run.config(state=tk.NORMAL), self.status_lbl.config(text="Sincronizzato.")])
        self.is_calculating = False

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson")])
        if path:
            self.gdf = gpd.read_file(path).to_crs(epsg=3857)
            self.gdf['id_hex'] = self.gdf.index.astype(str)
            self.tree = KDTree(np.array(list(zip(self.gdf.geometry.centroid.x, self.gdf.geometry.centroid.y))))
            self.btn_load_grid.config(text=f"✅ {os.path.basename(path)}")
            self.refresh_map()

    def load_ai(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.keras")])
        if p: self.model = load_model(p, compile=False); self.btn_load_ai.config(text=f"✅ {os.path.basename(p)}")

    def load_scaler(self):
        p = filedialog.askopenfilename(filetypes=[("Scaler", "*.gz")])
        if p: self.scaler = joblib.load(p); self.btn_load_scaler.config(text=f"✅ {os.path.basename(p)}")

    def show_profile_window(self):
        line = LineString(self.profile_pts)
        intersect = self.gdf[self.gdf.intersects(line)].copy()
        if intersect.empty: return
        intersect['dist'] = intersect.geometry.centroid.distance(Point(self.profile_pts[0]))
        intersect = intersect.sort_values('dist')
        win = tk.Toplevel(self.root); win.title("Profilo Altimetrico"); win.geometry("800x500")
        v_p = tk.StringVar(value="ip_dyn")
        cb = ttk.Combobox(win, textvariable=v_p, values=self.all_items, state="readonly"); cb.pack(pady=5)
        fig_p, ax_p = plt.subplots(figsize=(7, 4)); cv_p = FigureCanvasTkAgg(fig_p, master=win); cv_p.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        def upd(e=None):
            ax_p.clear(); v = v_p.get(); d = intersect['dist']/1000.0
            ax_p.plot(d, intersect[v], color='black', linewidth=2); ax_p.fill_between(d, 0, intersect[v], color='gray', alpha=0.3)
            ax_p.set_ylim(0, 1.1); ax_p.grid(True, alpha=0.3); ax_p.set_title(f"Profilo {v.upper()}"); cv_p.draw()
        cb.bind("<<ComboboxSelected>>", upd); upd()

    def show_lens_window(self, x, y):
        r_val = self.lens_radius.get()
        subset = self.gdf[self.gdf.intersects(Point(x, y).buffer(r_val))]
        if subset.empty: return
        means = subset[self.all_items].mean()
        win = tk.Toplevel(self.root); win.title(f"Lente R={int(r_val)}m"); win.geometry("500x850")
        fig_l, axes = plt.subplots(len(self.all_items), 1, figsize=(4, 12))
        plt.subplots_adjust(hspace=1.1, left=0.2, bottom=0.05)
        for i, var in enumerate(self.all_items):
            val = means[var]; ax = axes[i]; ax.set_xlim(0, 1); ax.set_yticks([])
            ax.set_xticks([0, 1]); ax.set_xticklabels(['0', '1'], fontsize=8, fontweight='bold')
            ax.set_title(f"{var.upper()}: {val:.3f}", fontsize=8, loc='center', fontweight='bold', pad=10)
            ax.imshow(np.array([[0, 1]]), cmap="RdYlGn", aspect='auto', extent=[0, 1, 0, 1], alpha=0.3)
            ax.axvline(val, color='black', linewidth=3, zorder=5)
        FigureCanvasTkAgg(fig_l, master=win).get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk(); app = ProximityNeuroDesigner(root); root.mainloop()
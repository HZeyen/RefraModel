"""
Main window class for the geological model builder application
"""
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer
import os
import copy
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from .ui.menu import Menu
from .ui.dialogs import PropertyEditorDialog, SelectPickfDialog
from .geometry.bodies import BodyManager
from .geometry.points import PointManager
from .geometry.lines import LineManager
from .inversion.forward_model import ForwardModel
from .inversion.inversion import Inversion
from .inversion.mesh_builder import MeshBuilder
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar


class ModelBuilder(QMainWindow):
    """Main window for the geological model builder"""
    
    def __init__(self, screens, xmin=0.0, xmax=100.0, ymin=-30.0, ymax=0.0, threshold_pct=1.0):
        super().__init__()
        self.setWindowTitle("Geological Model Builder")
        self.setGeometry(100, 100, 1500, 1000)
        
        # Initialize managers
        # Store domain and threshold
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.threshold_pct = threshold_pct

        self.body_manager = BodyManager()
        # Convert percentage threshold to absolute epsilons
        try:
            eps_x = abs((threshold_pct / 100.0) * (self.xmax - self.xmin))
            eps_y = abs((threshold_pct / 100.0) * (self.ymax - self.ymin))
        except Exception:
            eps_x, eps_y = 1.0, 1.0
        self.point_manager = PointManager(self.xmin, self.xmax, self.ymin, self.ymax, eps_x, eps_y)
        self.line_manager = LineManager()
        self.forward_model = ForwardModel()
        self.inversion = Inversion()
        self.mesh_builder = MeshBuilder()
        # Path to save model on commits
        self.model_save_path = None
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Plot mode state: False = interactive shown, True = inverted shown
        self.show_inverted = False
        self.show_rays = False  # Ray plot visibility state
        self.fig = None
        self.gs = None
        self.canvas = None
        self.toolbar = None
        self.ax_dat = None
        self.ax = None
        self.ax_cb = None
        # Picks / topography context
        self.scheme = None
        self.picks_flag = False
        self.xtopo = []
        self.ytopo = []
        self.topo_flag = False
        self.sht_nr = None
        self.pk_index = None
        self.pk_x = None
        # Node edit state
        self.node_edit_active = False
        self._drag_pid = None
        self._mpl_cids = []
        self._edit_highlights = []
        self._edit_backup = None
        self._snap_hint_artists = []
        self._snap_hint_timer = None
        self._edit_locked = False
        self._active_highlight_pid = None
        self._selected_pid = None  # For deletion
        # Body editor state
        self.body_edit_active = False
        self._body_edit_backup = None
        self._body_mpl_cids = []
        # Overlay message for edit modes
        self._overlay_message_text = None
        self._body_split_start = None  # (x, y) for drag start
        self._body_split_line = None   # Artist for preview line
        self._body_join_first = None   # First body index for join
        # Property editor state
        self.property_edit_active = False
        self._prop_mpl_cids = []
        self._prop_edit_backup = None
        self._body_split_last = None   # Last in-axes point during split drag
        
        # Create menu
        self.menu = Menu(self)
        self.setMenuBar(self.menu.menu_bar)
        
        # Additional setup
        self.setup_ui()
    
    def setup_ui(self):
        """Setup a single figure with GridSpec and one bottom toolbar."""
        # Single canvas for both top (data) and bottom (model+colorbar)
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Create GridSpec layout similar to the provided snippet
        self.gs = GridSpec(100, 100, figure=self.fig)
        self.ax_dat = self.fig.add_subplot(self.gs[:40, :95])
        self.ax = self.fig.add_subplot(self.gs[45:, :95])
        self.ax_cb = self.fig.add_subplot(self.gs[45:, 97:])

        # Initial titles/placeholders
        self.ax_dat.set_title("Data plot")
        self.ax.set_title("Interactive model")
        self.ax_cb.set_title("", pad=10)

        # Axis labels as requested
        self.ax.set_xlabel("Distance [m]")
        self.ax.set_ylabel("Depth [m]")
        self.ax_dat.set_ylabel("Time [s]")
        # Color scale styling on right side
        self.ax_cb.yaxis.set_label_position("right")
        self.ax_cb.yaxis.tick_right()
        self.ax_cb.set_ylabel("Velocity [m/s]")
        self.ax_cb.set_xticks([])

        # Tighter spacing between top and bottom axes
        self.fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.06, hspace=0.10, wspace=0.10)
        self.canvas.draw_idle()

        # Single toolbar at the very bottom
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

        # Optional status
        self.statusBar().showMessage("Ready")
    
    def keyPressEvent(self, event):
        """Handle key press events - mainly for body regularization mode."""
        from PyQt5.QtCore import Qt
        
        # Handle ENTER/ESC during body regularization mode
        if hasattr(self, '_body_reg_mode') and self._body_reg_mode:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                # Finish body regularization and run inversion
                self._finish_body_regularization(proceed=True)
            elif event.key() == Qt.Key_Escape:
                # Cancel body regularization
                self._finish_body_regularization(proceed=False)
        else:
            # Pass to parent for default handling
            super().keyPressEvent(event)
    
    def _finish_body_regularization(self, proceed=True):
        """Finish body regularization mode and optionally run inversion."""
        # Disconnect click handler
        if hasattr(self, '_body_reg_cid'):
            self.fig.canvas.mpl_disconnect(self._body_reg_cid)
            delattr(self, '_body_reg_cid')
        
        self._body_reg_mode = False
        
        if self.statusBar():
            self.statusBar().clearMessage()
        
        if proceed:
            # Run inversion with body regularization settings
            self.menu.run_inversion()
        else:
            # Clear body regularization settings
            if hasattr(self, 'body_regularization'):
                self.body_regularization = {}

    def toggle_model_plot_mode(self):
        """Toggle plot and update button without recreating widgets."""
        self.show_inverted = not self.show_inverted

        # Redraw the lower model axis to reflect current mode
        if self.ax is not None:
            self.ax.cla()
            if self.show_inverted:
                # Show inverted model if available
                if hasattr(self, 'inversion') and self.inversion and hasattr(self.inversion, 'vest') and self.inversion.vest is not None:
                    # Re-plot the inverted model
                    from ui.menu import Menu
                    if hasattr(self, 'menu') and self.menu:
                        self.menu._plot_inverted_model_to_canvas(self.inversion.vest, self.inversion.mgr, self)
                else:
                    self.ax.set_title("Inverted model (not available)")
                    self.ax.set_xlabel("Distance [m]")
                    self.ax.set_ylabel("Depth [m]")
            else:
                # Show interactive model - redraw bodies with colors
                self.ax.set_title("Interactive model")
                self.ax.set_xlabel("Distance [m]")
                self.ax.set_ylabel("Depth [m]")
                self.ax.set_xlim(self.xmin, self.xmax)
                self.ax.set_ylim(self.ymin, self.ymax)
                
                # Get velocity range across all bodies
                vmin, vmax = self.body_manager.get_vel_limits()
                
                # Create colormap and normalizer
                from matplotlib.colors import Normalize
                import matplotlib.cm as cm
                cmap = cm.Spectral_r
                norm = Normalize(vmin=vmin, vmax=vmax)
                
                # Draw all bodies as filled polygons
                for ib, body in enumerate(self.body_manager.bodies):
                    # Get polygon coordinates
                    x, y = self.body_manager.get_polygon(ib, self.point_manager.points, self.line_manager.lines)
                    
                    # Get body velocity (first property)
                    velocity = body["props"][0] if body["props"] else vmin
                    color = cmap(norm(velocity))
                    
                    # Create and add polygon patch
                    polygon = Polygon(list(zip(x, y)), facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
                    self.ax.add_patch(polygon)
        self.canvas.draw_idle()
    
    def toggle_ray_plot(self):
        """Toggle ray path visibility on the model plot."""
        self.show_rays = not self.show_rays
        
        # Remove existing ray artists if any
        if hasattr(self, '_ray_artists'):
            for artist in self._ray_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._ray_artists = []
        
        # Add rays if enabled
        if self.show_rays and self.ax is not None:
            self._ray_artists = []
            
            # Determine which rays to plot based on current view mode
            if self.show_inverted:
                # Plot inversion rays
                if hasattr(self, 'inversion') and self.inversion and hasattr(self.inversion, 'ray_paths'):
                    try:
                        for ray_x, ray_y in self.inversion.ray_paths:
                            line, = self.ax.plot(ray_x, ray_y, 'k-', lw=0.3, alpha=0.5)
                            self._ray_artists.append(line)
                    except Exception as e:
                        print(f"Could not draw inversion rays: {e}")
            else:
                # Plot forward model rays
                if hasattr(self, 'forward_model') and self.forward_model and hasattr(self.forward_model, 'ray_paths'):
                    try:
                        for ray_x, ray_y in self.forward_model.ray_paths:
                            line, = self.ax.plot(ray_x, ray_y, 'k-', lw=0.3, alpha=0.5)
                            self._ray_artists.append(line)
                    except Exception as e:
                        print(f"Could not draw forward model rays: {e}")
        
        self.canvas.draw_idle()
        
        # Update status bar
        if hasattr(self, 'statusBar'):
            state = "shown" if self.show_rays else "hidden"
            self.statusBar().showMessage(f"Ray paths {state}", 2000)
    
    def update_data_plot_title(self, chi2=None, rms=None):
        """Update data plot title with chi2 and RMS if available."""
        if self.ax_dat is not None:
            if chi2 is not None and rms is not None:
                self.ax_dat.set_title(f"Data plot (Chi² = {chi2:.3f}, RMS = {rms*1000:.2f} ms)")
            else:
                self.ax_dat.set_title("Data plot")
            self.canvas.draw_idle()
    
    def show_overlay_message(self, message):
        """Show a prominent overlay message on the data plot."""
        if self.ax_dat is None:
            return
        # Remove existing message if any
        self.hide_overlay_message()
        # Add new message in center of data plot
        self._overlay_message_text = self.ax_dat.text(
            0.5, 0.5, message,
            transform=self.ax_dat.transAxes,
            fontsize=16,
            fontweight='bold',
            color='red',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2)
        )
        self.canvas.draw_idle()
    
    def hide_overlay_message(self):
        """Remove the overlay message from the data plot."""
        if self._overlay_message_text is not None:
            try:
                self._overlay_message_text.remove()
            except Exception:
                pass
            self._overlay_message_text = None
            self.canvas.draw_idle()
    
    def plot_inverted_model(self):
        """Replot the inverted model with current color scale settings."""
        if not hasattr(self, 'inversion') or self.inversion is None or self.inversion.vest is None:
            return
        
        # Use the menu's plotting method to redraw with current color settings
        if hasattr(self, 'menu'):
            self.menu._plot_inverted_model_to_canvas(
                self.inversion.vest, 
                self.inversion.mgr, 
                self
            )

    # --- Model file loading (custom text format) ---
    def load_model_from_file(self, file_path: str):
        """Load model from a custom text file format.

        Format:
          Line 1: nbody, prop_name1, prop_name2, ...
          For each body:
            Header: npt, <prop values...>, name
            Then npt lines of "x,y" coordinates (last equals first)
        """
        # First pass: determine extents if not provided (xmin==xmax)
        if np.isclose(self.xmin, self.xmax):
            xmin = 1.0e7
            xmax = -1.0e7
            ymin = 1.0e7
            ymax = -1.0e7
            with open(file_path, "r") as fi:
                text = fi.readline()
                values = text.strip().split(",")
                nbody = int(values[0])
                for _ in range(nbody):
                    header = fi.readline()
                    hvals = header.strip().split(",")
                    npt = int(hvals[0])
                    for _ in range(npt):
                        line = fi.readline()
                        parts = line.strip().split(",")
                        if len(parts) < 2:
                            continue
                        # Normalize coordinates to 2 decimals
                        x = float(round(float(parts[0]), 2)); y = float(round(float(parts[1]), 2))
                        xmin = min(xmin, x); xmax = max(xmax, x)
                        ymin = min(ymin, y); ymax = max(ymax, y)
            self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
            self.point_manager.xmin = xmin
            self.point_manager.xmax = xmax
            self.point_manager.ymin = ymin
            self.point_manager.ymax = ymax
            xlen = abs(xmax - xmin)
            ylen = abs(ymax - ymin)
            self.point_manager.eps_x = (self.threshold_pct / 100.0) * xlen
            self.point_manager.eps_y = (self.threshold_pct / 100.0) * ylen

        # Second pass: construct bodies, points, and lines
        with open(file_path, "r") as fi:
            text = fi.readline()
            values = text.strip().split(",")
            nbody = int(values[0])
            prop_names = [v.lstrip() for v in values[1:]]

            # Reset managers
            self.body_manager.bodies = []
            self.body_manager.nbody = -1
            self.line_manager.lines = []
            self.line_manager.nline = -1
            self.point_manager.points = []
            self.point_manager.npoint = -1

            for ib in range(nbody):
                header = fi.readline()
                hvals = header.strip().split(",")
                npt = int(hvals[0])
                props = [float(v) for v in hvals[1:-1]] if len(hvals) > 2 else []
                name = hvals[-1].lstrip() if len(hvals) >= 2 else f"Body_{ib}"

                # Create body placeholder; lines and senses will be appended
                self.body_manager.append_body([], [], prop_names, para_values=props, name=name)

                # Collect/ensure unique points
                ptn = []  # indices of points in global list
                coords = []
                for _ in range(npt):
                    line = fi.readline()
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    # Normalize coordinates to 2 decimals
                    x = float(round(float(parts[0]), 2)); y = float(round(float(parts[1]), 2))
                    coords.append((x, y))
                    # Search for existing point by value
                    found_index = -1
                    for ipt, p in enumerate(self.point_manager.points):
                        if np.isclose(x, p["x"]) and np.isclose(y, p["y"]):
                            found_index = ipt
                            break
                    if found_index == -1:
                        # transform to screen for initial storage using normalized coords
                        xs, ys = self.ax.transData.transform((x, y)) if self.ax is not None else (x, y)
                        self.point_manager.append_point(x, y, xs, ys, body=[ib], line=[])
                        ptn.append(self.point_manager.npoint)
                    else:
                        ptn.append(found_index)

                # Build lines for polygon edges (consecutive pairs)
                for ip in range(1, len(ptn)):
                    pt1 = ptn[ip - 1]
                    pt2 = ptn[ip]
                    # Check if line already exists
                    existing = -1
                    reverse = False
                    for il, line in enumerate(self.line_manager.lines):
                        if line["point1"] == pt1 and line["point2"] == pt2:
                            existing = il; reverse = False; break
                        if line["point2"] == pt1 and line["point1"] == pt2:
                            existing = il; reverse = True; break
                    if existing == -1:
                        self.line_manager.append_line(pt1, pt2, [ib])
                        lin_idx = self.line_manager.nline
                    else:
                        lin_idx = existing
                        # ensure body membership tracked in line
                        if ib not in self.line_manager.lines[lin_idx]["bodies"]:
                            self.line_manager.lines[lin_idx]["bodies"].append(ib)
                    # Update body with line index and sense
                    sense = -1 if reverse else 1
                    self.body_manager.bodies[-1]["lines"].append(lin_idx)
                    self.body_manager.bodies[-1]["sense"].append(sense)
                    # Update point-line references
                    if lin_idx not in self.point_manager.points[pt1]["lines"]:
                        self.point_manager.points[pt1]["lines"].append(lin_idx)
                    if lin_idx not in self.point_manager.points[pt2]["lines"]:
                        self.point_manager.points[pt2]["lines"].append(lin_idx)

        # Adjust axes limits to model extents
        if self.ax is not None:
            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.ymin, self.ymax)
            self.canvas.draw_idle()
        
        # Plot the loaded model
        self.plot_model()

    def start_model(self, prop_names=None, prop_values=None):
        """Create a simple rectangular starting model with bounds (xmin,xmax) x (ymin,ymax).

        Integrates 4 corner points, 4 edges, and 1 body into the same dictionary structure.
        
        Parameters
        ----------
        prop_names : list of str, optional
            Names of properties for the body. Default: ["velocity"]
        prop_values : list of float, optional
            Values of properties for the body. Default: [1500.0]
        """
        if prop_names is None:
            prop_names = ["velocity"]
        if prop_values is None:
            prop_values = [1500.0]
        
        # Reset managers
        self.point_manager.points = []
        self.point_manager.npoint = -1
        self.line_manager.lines = []
        self.line_manager.nline = -1
        self.body_manager.bodies = []
        self.body_manager.nbody = -1

        # Transform corners to screen coordinates
        if self.ax is not None:
            screen_xmin, screen_ymax = self.ax.transData.transform((self.xmin, self.ymax))
            screen_xmax, _ = self.ax.transData.transform((self.xmax, self.ymax))
            _, screen_ymin = self.ax.transData.transform((self.xmin, self.ymin))
        else:
            screen_xmin, screen_ymax = self.xmin, self.ymax
            screen_xmax, screen_ymin = self.xmax, self.ymin

        # Append 4 corner points: top-left, top-right, bottom-right, bottom-left
        self._append_point_extended(self.xmin, self.ymax, screen_xmin, screen_ymax, body=[0], line=[0, 3])
        self._append_point_extended(self.xmax, self.ymax, screen_xmax, screen_ymax, body=[0], line=[0, 1])
        self._append_point_extended(self.xmax, self.ymin, screen_xmax, screen_ymin, body=[0], line=[1, 2])
        self._append_point_extended(self.xmin, self.ymin, screen_xmin, screen_ymin, body=[0], line=[2, 3])

        # Append 4 lines: top, right, bottom, left
        self._append_line_extended(0, 1, [0])
        self._append_line_extended(1, 2, [0])
        self._append_line_extended(2, 3, [0])
        self._append_line_extended(3, 0, [0])

        # Append 1 body with all 4 lines (sense=1 for all)
        self.body_manager.append_body([0, 1, 2, 3], [1, 1, 1, 1], prop_names, para_values=prop_values, name="Base")

        # Adjust axes limits
        if self.ax is not None:
            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.ymin, self.ymax)
            self.canvas.draw_idle()
        
        # Plot the starting model
        self.plot_model()

    def _append_point_extended(self, xp, yp, xs, ys, body=None, line=None):
        """Append a point with extended flags (topo, bottom, left, right)."""
        body = body or []
        line = line or []
        self.point_manager.npoint += 1
        # Round coordinates to 2 decimals
        try:
            xp = float(round(xp, 2))
            yp = float(round(yp, 2))
        except Exception:
            pass
        # Recompute screen coords from rounded values when possible
        try:
            if self.ax is not None:
                xs, ys = self.ax.transData.transform((xp, yp))
        except Exception:
            pass
        yt = self._get_topo(xp)
        point = {
            "x": xp,
            "y": yp,
            "xscreen": xs,
            "yscreen": ys,
            "lines": line,
            "bodies": body,
            "topo": np.isclose(yt, yp),
            "bottom": np.isclose(yp, self.ymin),
            "left": xp <= self.xmin + 0.1,
            "right": xp >= self.xmax - 0.01,
        }
        self.point_manager.points.append(point)

    def _append_line_extended(self, pt1, pt2, body=None):
        """Append a line with extended flags (topo, bottom, left, right)."""
        body = body or []
        self.line_manager.nline += 1
        p1 = self.point_manager.points[pt1]
        p2 = self.point_manager.points[pt2]
        line = {
            "point1": pt1,
            "point2": pt2,
            "bodies": body,
            "topo": p1.get("topo", False) and p2.get("topo", False),
            "bottom": p1.get("bottom", False) and p2.get("bottom", False),
            "left": p1.get("left", False) and p2.get("left", False),
            "right": p1.get("right", False) and p2.get("right", False),
        }
        self.line_manager.lines.append(line)

    def _get_topo(self, xp):
        """Return topography y at x=xp. Stub: returns ymax (flat surface)."""
        yt = self._topo_y(xp)
        return yt if yt is not None else self.ymax

    def _topo_y(self, x):
        """Interpolate topography y at coordinate x if xtopo/ytopo available."""
        try:
            if isinstance(self.xtopo, (list, np.ndarray)) and isinstance(self.ytopo, (list, np.ndarray)):
                if len(self.xtopo) >= 2 and len(self.xtopo) == len(self.ytopo):
                    xt = np.array(self.xtopo, dtype=float)
                    yt = np.array(self.ytopo, dtype=float)
                    order = np.argsort(xt)
                    xt = xt[order]; yt = yt[order]
                    # Clamp outside range to nearest end
                    if x <= xt[0]:
                        return float(yt[0])
                    if x >= xt[-1]:
                        return float(yt[-1])
                    return float(np.interp(x, xt, yt))
        except Exception:
            pass
        return None

    def plot_model(self):
        """Plot all bodies in the lower-left axis colored by velocity, and show colorbar in lower-right axis."""
        if self.ax is None or not self.body_manager.bodies:
            return
        
        # Clear the model axis
        self.ax.cla()
        self.ax.set_xlabel("Distance [m]")
        self.ax.set_ylabel("Depth [m]")
        self.ax.set_title("Interactive model")
        
        # Get velocity range across all bodies
        vmin, vmax = self.body_manager.get_vel_limits()
        
        # Create colormap and normalizer
        cmap = cm.Spectral_r
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Plot each body as a polygon
        for ib, body in enumerate(self.body_manager.bodies):
            # Get polygon coordinates
            x, y = self.body_manager.get_polygon(ib, self.point_manager.points, self.line_manager.lines)
            
            # Get body velocity (first property)
            velocity = body["props"][0] if body["props"] else vmin
            color = cmap(norm(velocity))
            
            # Create and add polygon patch
            polygon = Polygon(list(zip(x, y)), facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            self.ax.add_patch(polygon)
        
        # Set axis limits - include sensor positions if available
        xmin_plot = self.xmin
        xmax_plot = self.xmax
        
        if self.scheme is not None:
            try:
                sensors = self.scheme.sensors().array()
                xmin_plot = min(xmin_plot, np.min(sensors[:, 0]))
                xmax_plot = max(xmax_plot, np.max(sensors[:, 0]))
            except Exception:
                pass
        
        self.ax.set_xlim(xmin_plot, xmax_plot)
        self.ax.set_ylim(self.ymin, self.ymax)
        # Don't use equal aspect - it changes xlim. Let matplotlib use the axis box size.
        # self.ax.set_aspect('equal', adjustable='box')
        
        # Update colorbar in the right axis
        self.ax_cb.cla()
        self.ax_cb.yaxis.set_label_position("right")
        self.ax_cb.yaxis.tick_right()
        self.ax_cb.set_ylabel("Velocity [m/s]")
        self.ax_cb.set_xticks([])
        
        # Draw colorbar
        cb = ColorbarBase(self.ax_cb, cmap=cmap, norm=norm, orientation='vertical')
        
        # Draw surface/topography line if available
        if isinstance(self.xtopo, (list, np.ndarray)) and len(self.xtopo) > 1 and isinstance(self.ytopo, (list, np.ndarray)) and len(self.ytopo) == len(self.xtopo):
            try:
                self.ax.plot(self.xtopo, self.ytopo, color='k', linewidth=1.0, zorder=4)
            except Exception:
                pass

        # Plot body edge points as light grey ellipses with threshold radii
        try:
            rx = (self.threshold_pct / 100.0) * abs(self.xmax - self.xmin)
            ry = (self.threshold_pct / 100.0) * abs(self.ymax - self.ymin)
            # Collect unique edge point indices from lines that belong to bodies
            edge_points = set()
            for line in self.line_manager.lines:
                if line.get("bodies"):
                    edge_points.add(line["point1"])
                    edge_points.add(line["point2"])
            for ip in edge_points:
                p = self.point_manager.points[ip]
                e = Ellipse((p["x"], p["y"]), width=2*rx, height=2*ry,
                            facecolor="#a9a9a9", edgecolor='none', alpha=0.8, zorder=3)
                self.ax.add_patch(e)
        except Exception:
            pass

        # Redraw canvas
        self.canvas.draw_idle()
        # Restore persistent highlights while in edit mode
        self._restore_highlights()
        

    def integrate_topography_into_surface(self):
        """Integrate xtopo/ytopo points as the uppermost boundary of surface body.

        Currently supported for the simple single-body start model by
        replacing its top edge with segments following (xtopo, ytopo).
        """
        if not (isinstance(self.xtopo, (list, np.ndarray)) and isinstance(self.ytopo, (list, np.ndarray))):
            return
        if len(self.xtopo) < 2 or len(self.xtopo) != len(self.ytopo):
            return
        # Only handle simple single-body case for now
        if len(self.body_manager.bodies) != 1:
            return
        # Identify side and bottom lines from flags
        side_left = side_right = bottom = None
        top_candidates = []
        for il, line in enumerate(self.line_manager.lines):
            if line.get("left"):
                side_left = il
            elif line.get("right"):
                side_right = il
            elif line.get("bottom"):
                bottom = il
            else:
                top_candidates.append(il)
        # Build top points from topo
        new_top_pts = []
        try:
            xt = np.array(self.xtopo, dtype=float)
            yt = np.array(self.ytopo, dtype=float)
            order = np.argsort(xt)
            xt = xt[order]; yt = yt[order]
        except Exception:
            return
        for x, y in zip(xt, yt):
            try:
                xs, ys = self.ax.transData.transform((x, y)) if self.ax is not None else (x, y)
                self._append_point_extended(x, y, xs, ys, body=[0], line=[])
                # Mark as original topography point (cannot move)
                self.point_manager.points[self.point_manager.npoint]["topo_original"] = True
                new_top_pts.append(self.point_manager.npoint)
            except Exception:
                pass
        if len(new_top_pts) < 2:
            return
        # Create new top line segments connecting new_top_pts
        new_top_lines = []
        for i in range(1, len(new_top_pts)):
            pt1 = new_top_pts[i-1]
            pt2 = new_top_pts[i]
            self._append_line_extended(pt1, pt2, [0])
            new_top_lines.append(self.line_manager.nline)
        # Mark these new lines as topography
        for il in new_top_lines:
            self.line_manager.lines[il]["topo"] = True
        # Update body 0 to use the new top chain followed by right, bottom, left
        body = self.body_manager.bodies[0]
        updated_lines = list(new_top_lines)
        if side_right is not None:
            updated_lines.append(side_right)
        if bottom is not None:
            updated_lines.append(bottom)
        if side_left is not None:
            updated_lines.append(side_left)
        body["lines"] = updated_lines
        body["sense"] = [1] * len(updated_lines)
        # Update point-line references for the corner/top points
        for il in new_top_lines:
            line = self.line_manager.lines[il]
            p1 = line["point1"]; p2 = line["point2"]
            if il not in self.point_manager.points[p1]["lines"]:
                self.point_manager.points[p1]["lines"].append(il)
            if il not in self.point_manager.points[p2]["lines"]:
                self.point_manager.points[p2]["lines"].append(il)

    # --- Picks handling ---
    def set_scheme(self, scheme):
        """Attach a pygimli traveltime scheme, derive pick positions and optionally topography."""
        self.scheme = scheme
        self.picks_flag = True
        # Derive pick positions
        try:
            self.get_pick_positions()
        except Exception:
            pass
        # Derive topography from sensors if z varies
        try:
            pos = np.array(self.scheme.sensors())
            if pos.shape[1] >= 3:
                t = np.unique(pos[:, 2])
                if len(t) > 1:
                    self.xtopo = list(pos[:, 0])
                    self.ytopo = list(pos[:, 2])
                    xx, index = np.unique(np.array(self.xtopo), return_index=True)
                    if len(xx) > 1:
                        self.xtopo = np.copy(xx)
                        self.ytopo = np.copy(np.array(self.ytopo)[index])
                        self.topo_flag = True
        except Exception:
            pass

    def get_pick_positions(self):
        """Compute pick grouping indices and geophone x-positions from scheme."""
        if self.scheme is None:
            return
        self.sht_nr, self.pk_index = np.unique(self.scheme["s"], return_index=True)
        self.pk_index = np.insert(self.pk_index, len(self.pk_index), len(self.scheme["s"]))
        pk_pos = np.array(self.scheme.sensors())[self.scheme["g"], :]
        self.pk_x = pk_pos[:, 0]

    def plot_picks(self, ax, data, marker=None):
        """Plot measured travel times as picks in the provided axis."""
        if self.scheme is None or self.pk_index is None or self.pk_x is None:
            return
        ax.set_prop_cycle(None)
        for ipos in range(len(self.pk_index) - 1):
            i1 = int(self.pk_index[ipos])
            i2 = int(self.pk_index[ipos + 1])
            x = self.pk_x[i1:i2]
            y = data[i1:i2]
            try:
                err = self.scheme["err"][i1:i2]
            except Exception:
                err = np.zeros_like(y)
            sht = int(self.scheme["s"][i1])
            try:
                xsht = self.scheme.sensors().array()[sht, 0]
            except Exception:
                xsht = self.scheme.sensors()[sht][0]
            index = np.where(x > xsht)[0]
            index = int(index[0]) if len(index) > 0 else len(x)
            x = np.insert(x, index, xsht)
            y = np.insert(y, index, 0.0)
            err = np.insert(err, index, 0.0001)
            if marker is None:
                ax.plot(x, y)
            else:
                ax.errorbar(x, y, yerr=err, fmt=marker)
        
        # Set xlim to match full extent (sensors + model range)
        try:
            xmin_plot = self.xmin
            xmax_plot = self.xmax
            
            # Also consider sensor positions if scheme exists
            if self.scheme is not None:
                sensors = self.scheme.sensors().array()
                xmin_plot = min(xmin_plot, np.min(sensors[:, 0]))
                xmax_plot = max(xmax_plot, np.max(sensors[:, 0]))
            
            ax.set_xlim(xmin_plot, xmax_plot)
        except Exception:
            pass
        
        self.canvas.draw_idle()

    def plot_calculated_times(self, response):
        """Plot calculated travel times as continuous lines with same colors as measured picks."""
        if self.scheme is None or self.pk_index is None or self.pk_x is None:
            return
        
        # Convert response to numpy array if needed
        if not isinstance(response, np.ndarray):
            response = np.array(response)
        
        # Clear any previous calculated time lines
        if hasattr(self, '_calc_time_lines'):
            for line in self._calc_time_lines:
                try:
                    line.remove()
                except:
                    pass
        self._calc_time_lines = []
        
        # Get the color cycle to match pick colors
        self.ax_dat.set_prop_cycle(None)
        
        # Plot calculated times for each shot with matching colors
        for ipos in range(len(self.pk_index) - 1):
            i1 = int(self.pk_index[ipos])
            i2 = int(self.pk_index[ipos + 1])
            x = self.pk_x[i1:i2].copy()
            y_calc = response[i1:i2].copy()
            
            # Get shot position (use same method as plot_picks)
            sht = int(self.scheme["s"][i1])
            try:
                xsht = self.scheme.sensors().array()[sht, 0]
            except Exception:
                xsht = self.scheme.sensors()[sht][0]
            
            # Insert shot position at appropriate index
            index = np.where(x > xsht)[0]
            index = int(index[0]) if len(index) > 0 else len(x)
            x = np.insert(x, index, xsht)
            y_calc = np.insert(y_calc, index, 0.0)
            
            # Plot as continuous line (no marker, just line)
            line, = self.ax_dat.plot(x, y_calc, '-', linewidth=1.5)
            self._calc_time_lines.append(line)
        
        self.canvas.draw_idle()

    # --- Node editing mode ---
    def start_node_edit_mode(self):
        # Block if another edit mode is active
        if self.body_edit_active or self.property_edit_active:
            mode = self._active_edit_mode_name()
            try:
                QMessageBox.information(self, "Finish Edit Mode",
                                        f"Finish {mode} edit with Enter or Esc before continuing.")
            except Exception:
                self.statusBar().showMessage(f"Finish {mode} edit with Enter or Esc before continuing", 4000)
            return
        if self.node_edit_active:
            return
        self.node_edit_active = True
        self.statusBar().showMessage("Node edit: Right-click add, Left-drag move, Left-click+DEL delete, Enter commit, Esc undo all")
        self.show_overlay_message("NODE EDIT MODE\nPress ENTER to save | Press ESC to cancel")
        # Backup current geometry for ESC undo
        try:
            self._edit_backup = {
                "points": copy.deepcopy(self.point_manager.points),
                "lines": copy.deepcopy(self.line_manager.lines),
                "bodies": copy.deepcopy(self.body_manager.bodies),
            }
        except Exception:
            self._edit_backup = None
        # Ensure canvas has keyboard focus and disable toolbar interactions during edit
        try:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
        except Exception:
            pass
        try:
            if self.toolbar is not None:
                self.toolbar.setEnabled(False)
        except Exception:
            pass
        self._edit_locked = False  # Allow multiple sequential edits
        self._active_highlight_pid = None
        self._selected_pid = None
        # Refresh screen coords to current transform
        self._refresh_screen_coords()
        # Connect mpl events
        self._mpl_cids = [
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press),
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move),
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release),
            self.canvas.mpl_connect('key_press_event', self._on_key_press),
        ]

    def start_body_edit_mode(self):
        # Block if another edit mode is active
        if self.node_edit_active or self.property_edit_active:
            mode = self._active_edit_mode_name()
            try:
                QMessageBox.information(self, "Finish Edit Mode",
                                        f"Finish {mode} edit with Enter or Esc before continuing.")
            except Exception:
                self.statusBar().showMessage(f"Finish {mode} edit with Enter or Esc before continuing", 4000)
            return
        if self.body_edit_active:
            return
        self.body_edit_active = True
        self.statusBar().showMessage("Body edit: Left-drag to split bodies, Right-click two adjacent bodies to join, Enter to commit, Esc to undo")
        self.show_overlay_message("BODY EDIT MODE\nPress ENTER to save | Press ESC to cancel")
        # Backup current geometry
        try:
            self._body_edit_backup = {
                "points": copy.deepcopy(self.point_manager.points),
                "lines": copy.deepcopy(self.line_manager.lines),
                "bodies": copy.deepcopy(self.body_manager.bodies),
            }
        except Exception:
            self._body_edit_backup = None
        # Ensure canvas focus
        try:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
        except Exception:
            pass
        try:
            if self.toolbar is not None:
                self.toolbar.setEnabled(False)
        except Exception:
            pass
        self._body_split_start = None
        self._body_split_line = None
        self._body_join_first = None
        self._body_split_last = None
        self._refresh_screen_coords()
        # Connect mpl events
        self._body_mpl_cids = [
            self.canvas.mpl_connect('button_press_event', self._on_body_mouse_press),
            self.canvas.mpl_connect('motion_notify_event', self._on_body_mouse_move),
            self.canvas.mpl_connect('button_release_event', self._on_body_mouse_release),
            self.canvas.mpl_connect('key_press_event', self._on_body_key_press),
        ]

    def stop_body_edit_mode(self):
        if not self.body_edit_active:
            return
        for cid in self._body_mpl_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._body_mpl_cids = []
        if self._body_split_line is not None:
            try:
                self._body_split_line.remove()
            except Exception:
                pass
            self._body_split_line = None
        self.body_edit_active = False
        self._body_edit_backup = None
        self._body_split_start = None
        self._body_join_first = None
        try:
            if self.toolbar is not None:
                self.toolbar.setEnabled(True)
        except Exception:
            pass
        self.hide_overlay_message()
        self.statusBar().showMessage("Ready")

    def stop_node_edit_mode(self):
        if not self.node_edit_active:
            return
        for cid in self._mpl_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids = []
        self._clear_highlights()
        self.node_edit_active = False
        self._drag_pid = None
        self._edit_backup = None
        # Re-enable toolbar after editing
        try:
            if self.toolbar is not None:
                self.toolbar.setEnabled(True)
        except Exception:
            pass
        self.hide_overlay_message()
        self.statusBar().showMessage("Ready")

    # --- Property editor mode ---
    def start_property_edit_mode(self):
        # Block if another edit mode is active
        if self.node_edit_active or self.body_edit_active:
            mode = self._active_edit_mode_name()
            try:
                QMessageBox.information(self, "Finish Edit Mode",
                                        f"Finish {mode} edit with Enter or Esc before continuing.")
            except Exception:
                self.statusBar().showMessage(f"Finish {mode} edit with Enter or Esc before continuing", 4000)
            return
        if self.property_edit_active:
            return
        self.property_edit_active = True
        self.statusBar().showMessage("Property editor: Click bodies to edit. Enter=accept, Esc=cancel.", 4000)
        self.show_overlay_message("PROPERTY EDIT MODE\nPress ENTER to save | Press ESC to cancel")
        # Backup current bodies (for Esc cancel)
        try:
            self._prop_edit_backup = copy.deepcopy(self.body_manager.bodies)
        except Exception:
            self._prop_edit_backup = None
        try:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
        except Exception:
            pass
        try:
            if self.toolbar is not None:
                self.toolbar.setEnabled(False)
        except Exception:
            pass
        # Connect to minimal events: click and key press
        self._prop_mpl_cids = [
            self.canvas.mpl_connect('button_press_event', self._on_prop_mouse_press),
            self.canvas.mpl_connect('key_press_event', self._on_prop_key_press),
        ]
        

    def stop_property_edit_mode(self):
        if not self.property_edit_active:
            return
        for cid in self._prop_mpl_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._prop_mpl_cids = []
        self.property_edit_active = False
        try:
            if self.toolbar is not None:
                self.toolbar.setEnabled(True)
        except Exception:
            pass
        self.hide_overlay_message()
        self.statusBar().showMessage("Ready")

    def _save_model_if_configured(self):
        """Save current model to configured path, if set."""
        if not self.model_save_path:
            return
        try:
            self.save_model_to_file(self.model_save_path)
            self.statusBar().showMessage(f"Model saved to {self.model_save_path}", 3000)
        except Exception as e:
            try:
                QMessageBox.warning(self, "Save Model", f"Failed to save model: {e}")
            except Exception:
                pass

    def save_model_to_file(self, filename: str):
        """Save model as CSV-like text per requested structure.

        First line: number_of_bodies, then parameter names (comma-separated)
        For each body:
          - One line: number_of_nodes, then parameter values
          - Then one line per node: x, y
        The last node equals the first.
        """
        bodies = self.body_manager.bodies
        n_bodies = len(bodies)
        # Determine parameter names from first body if available
        param_names = []
        if n_bodies > 0:
            param_names = list(bodies[0].get("prop_names", []))
        with open(filename, "w") as f:
            # Header line
            header_parts = [str(n_bodies)] + [str(p) for p in param_names]
            f.write(",".join(header_parts) + "\n")
            # Body blocks
            for ib in range(n_bodies):
                body = bodies[ib]
                x, y = self.body_manager.get_polygon(ib, self.point_manager.points, self.line_manager.lines)
                # Ensure closure
                if not (len(x) >= 2 and np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
                    x = list(x)
                    y = list(y)
                    x.append(x[0])
                    y.append(y[0])
                n_nodes = len(x)
                props = body.get("props", [])
                body_name = body.get("name", "")
                line_parts = [str(n_nodes)] + [f"{v}" for v in props] + [body_name]
                f.write(",".join(line_parts) + "\n")
                for xi, yi in zip(x, y):
                    try:
                        f.write(f"{float(xi):.2f},{float(yi):.2f}\n")
                    except Exception:
                        f.write(f"{xi},{yi}\n")

    def _active_edit_mode_name(self):
        if self.node_edit_active:
            return "node"
        if self.body_edit_active:
            return "body"
        if self.property_edit_active:
            return "property"
        return "edit"

    def _on_prop_key_press(self, event):
        if not self.property_edit_active:
            return
        if event.key in ('escape',):
            # Restore backup and exit
            try:
                if isinstance(self._prop_edit_backup, list):
                    self.body_manager.bodies = copy.deepcopy(self._prop_edit_backup)
                    self.body_manager.nbody = len(self.body_manager.bodies) - 1
                    self.plot_model()
            except Exception:
                pass
            self.stop_property_edit_mode()
        elif event.key in ('enter', 'return'):
            # Accept current changes and exit
            self._save_model_if_configured()
            self.stop_property_edit_mode()

    def _on_prop_mouse_press(self, event):
        if not self.property_edit_active:
            return
        if event.button != 1:
            return
        # Accept outside-axes clicks by transforming to data and clamping
        if event.xdata is None or event.ydata is None:
            try:
                xd, yd = self.ax.transData.inverted().transform((event.x, event.y))
            except Exception:
                return
            xq, yq = self._clamp_to_domain(xd, yd)
        else:
            xq, yq = float(event.xdata), float(event.ydata)
        bid, _ = self.body_manager.inside_body(xq, yq, self.point_manager.points, self.line_manager.lines)
        if bid < 0:
            self.statusBar().showMessage("No body at click location", 2000)
            return
        body = self.body_manager.bodies[bid]
        dlg = PropertyEditorDialog(
            body_index=bid,
            body_name=body.get("name", f"Body_{bid}"),
            prop_names=body.get("prop_names", []),
            prop_values=body.get("props", []),
            parent=self,
        )
        if dlg.exec_() == dlg.Accepted:
            name, values = dlg.values()
            if name is not None and values is not None:
                try:
                    body["name"] = name or body.get("name", f"Body_{bid}")
                    # Ensure props list length matches
                    pn = body.get("prop_names", [])
                    if len(values) != len(pn):
                        # Resize to match names count
                        if len(values) < len(pn):
                            values = values + [0.0] * (len(pn) - len(values))
                        else:
                            values = values[:len(pn)]
                    body["props"] = values
                    self.plot_model()
                    self.statusBar().showMessage(f"Updated body {bid}. Continue editing or press Enter/Esc.", 3000)
                except Exception:
                    self.statusBar().showMessage("Failed to update properties", 3000)
        # Keep mode active for multiple edits; finish with Enter/Esc

    

    def _on_key_press(self, event):
        if not self.node_edit_active:
            return
        if event.key in ('enter', 'return'):
            self._save_model_if_configured()
            self.stop_node_edit_mode()
        elif event.key in ('delete', 'backspace'):
            # Delete the selected point if not an original topo point
            if self._selected_pid is not None:
                self._delete_point(self._selected_pid)
        elif event.key == 'escape':
            # Restore backup and exit edit mode
            if isinstance(self._edit_backup, dict):
                try:
                    self.point_manager.points = copy.deepcopy(self._edit_backup.get("points", []))
                    self.line_manager.lines = copy.deepcopy(self._edit_backup.get("lines", []))
                    self.body_manager.bodies = copy.deepcopy(self._edit_backup.get("bodies", []))
                    self.point_manager.npoint = len(self.point_manager.points) - 1
                    self.line_manager.nline = len(self.line_manager.lines) - 1
                    self.body_manager.nbody = len(self.body_manager.bodies) - 1
                    self._refresh_screen_coords()
                    self.plot_model()
                except Exception:
                    pass
            self.stop_node_edit_mode()

    def _on_mouse_press(self, event):
        if not self.node_edit_active or event.inaxes != self.ax:
            return
        # Left button: select/start dragging nearest point (in screen coordinates)
        if event.button == 1:
            pid = self._nearest_point_screen(event.x, event.y)
            if pid is not None:
                # Check if it's an original topography point (cannot move)
                p = self.point_manager.points[pid]
                if p.get("topo_original", False):
                    self._selected_pid = pid  # Can select for delete check, but no drag
                    self.statusBar().showMessage("Original topography point: cannot move (only delete added points)", 2000)
                    return
                self._drag_pid = pid
                self._selected_pid = pid
                # Redraw model and highlight connected lines in red
                self.plot_model()
                self._highlight_point_lines(pid)
                self._active_highlight_pid = pid
        # Right button: attempt to insert a point on nearest line (unless inside a point ellipse)
        elif event.button == 3:
            # Ignore if within any point ellipse (in data coordinates)
            if self._inside_any_point_ellipse(event):
                return
            # Find nearest line in screen space
            lidx = self.line_manager.find_nearest_line(event.x, event.y, self.point_manager.points)
            if lidx == -1:
                return
            # Compute projection point in screen coords via crossing again
            xp, yp, ok = self.line_manager.crossing(
                self.point_manager.points[self.line_manager.lines[lidx]["point1"]]["xscreen"],
                self.point_manager.points[self.line_manager.lines[lidx]["point1"]]["yscreen"],
                self.point_manager.points[self.line_manager.lines[lidx]["point2"]]["xscreen"],
                self.point_manager.points[self.line_manager.lines[lidx]["point2"]]["yscreen"],
                event.x, event.y,
            )
            if not ok:
                return
            # Transform to data coords
            try:
                xdata, ydata = self.ax.transData.inverted().transform((xp, yp))
            except Exception:
                return
            # Snap to topography/edges using tolerances
            xdata, ydata, snapped, kind = self._apply_snapping(xdata, ydata, exclude_pid=None)
            # Round inserted coordinates to 2 decimals before computing screen coords
            try:
                xdata = float(round(xdata, 2))
                ydata = float(round(ydata, 2))
            except Exception:
                pass
            # Recompute screen coords after snapping
            xs, ys = self.ax.transData.transform((xdata, ydata))
            # Append point and split line
            pnew = self._split_line_with_point(lidx, xdata, ydata, xs, ys)
            self.plot_model()
            if snapped:
                self._show_snap_hint(kind, xdata, ydata)
            # Persist highlight on new point (no lock, multi-edit allowed)
            if pnew is not None:
                self._active_highlight_pid = pnew
                self._selected_pid = pnew
                self._highlight_point_lines(pnew)

    def _on_mouse_move(self, event):
        if not self.node_edit_active or self._drag_pid is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Update point position in data with snapping
        pid = self._drag_pid
        p = self.point_manager.points[pid]
        # Check if point is on topography (but not original)
        if p.get("topo", False) and not p.get("topo_original", False):
            # Allow only X movement; Y follows interpolated topography
            xnew = float(event.xdata)
            # Snap X to edges/nodes if needed
            xnew, _, snapped_x, kind_x = self._apply_snapping(xnew, p["y"], exclude_pid=self._drag_pid)
            # Recompute Y from topography
            ynew = self._topo_y(xnew)
            if ynew is None:
                ynew = p["y"]  # fallback
            snapped = snapped_x
            kind = kind_x if snapped_x else 'topo'
        else:
            # Normal snapping
            xnew, ynew, snapped, kind = self._apply_snapping(float(event.xdata), float(event.ydata), exclude_pid=self._drag_pid)
        # Round moved coordinates to 2 decimals
        try:
            xnew = float(round(xnew, 2))
            ynew = float(round(ynew, 2))
        except Exception:
            pass
        self.point_manager.points[pid]["x"] = xnew
        self.point_manager.points[pid]["y"] = ynew
        xs, ys = self.ax.transData.transform((xnew, ynew))
        self.point_manager.points[pid]["xscreen"] = xs
        self.point_manager.points[pid]["yscreen"] = ys
        # Redraw model first, then update highlights as the point moves
        self.plot_model()
        if snapped:
            self._show_snap_hint(kind, xnew, ynew)
        self._highlight_point_lines(pid)

    def _on_mouse_release(self, event):
        if not self.node_edit_active:
            return
        if event.button == 1 and self._drag_pid is not None:
            self._drag_pid = None
            self.plot_model()

    # --- Helpers ---
    def _refresh_screen_coords(self):
        if self.ax is None:
            return
        for p in self.point_manager.points:
            x, y = p["x"], p["y"]
            xs, ys = self.ax.transData.transform((x, y))
            p["xscreen"], p["yscreen"] = xs, ys

    def _nearest_point_screen(self, xs, ys):
        if not self.point_manager.points:
            return None
        dmin = 1e18
        pid = None
        for i, p in enumerate(self.point_manager.points):
            dx = p["xscreen"] - xs
            dy = p["yscreen"] - ys
            d = dx*dx + dy*dy
            if d < dmin:
                dmin = d
                pid = i
        return pid

    def _inside_any_point_ellipse(self, event):
        if event.xdata is None or event.ydata is None:
            return False
        rx = (self.threshold_pct / 100.0) * abs(self.xmax - self.xmin)
        ry = (self.threshold_pct / 100.0) * abs(self.ymax - self.ymin)
        if rx <= 0 or ry <= 0:
            return False
        ex = float(event.xdata)
        ey = float(event.ydata)
        for p in self.point_manager.points:
            dx = (ex - p["x"]) / rx
            dy = (ey - p["y"]) / ry
            if dx*dx + dy*dy <= 1.0:
                return True
        return False

    def _split_line_with_point(self, lin_idx, xdata, ydata, xs, ys):
        # Append new point
        self.point_manager.append_point(xdata, ydata, xs, ys, body=[], line=[])
        pnew = self.point_manager.npoint
        line = self.line_manager.lines[lin_idx]
        a = line["point1"]; b = line["point2"]
        # Modify existing line to (a -> pnew)
        line["point2"] = pnew
        # Append new line (pnew -> b)
        self.line_manager.append_line(pnew, b, body=line.get("bodies", []).copy())
        new_lin_idx = self.line_manager.nline
        # Update point-line references
        if lin_idx not in self.point_manager.points[a]["lines"]:
            self.point_manager.points[a]["lines"].append(lin_idx)
        if lin_idx not in self.point_manager.points[pnew]["lines"]:
            self.point_manager.points[pnew]["lines"].append(lin_idx)
        self.point_manager.points[pnew]["lines"].append(new_lin_idx)
        self.point_manager.points[b]["lines"].append(new_lin_idx)
        # Update bodies that reference the old line
        for ib, body in enumerate(self.body_manager.bodies):
            j = 0
            while j < len(body["lines"]):
                if body["lines"][j] == lin_idx:
                    s = body["sense"][j]
                    if s >= 0:
                        # Insert new line after current
                        body["lines"].insert(j+1, new_lin_idx)
                        body["sense"].insert(j+1, +1)
                        j += 2
                    else:
                        # Insert new line before current for reversed order
                        body["lines"].insert(j, new_lin_idx)
                        body["sense"].insert(j, -1)
                        j += 2
                else:
                    j += 1
        return pnew

    def _apply_snapping(self, x, y, exclude_pid=None):
        """Snap a coordinate to topography or model boundaries within eps tolerances.
        Returns (x, y, snapped_flag, snap_kind)."""
        try:
            eps_x = float(getattr(self.point_manager, 'eps_x', 0.0) or 0.0)
            eps_y = float(getattr(self.point_manager, 'eps_y', 0.0) or 0.0)
        except Exception:
            eps_x, eps_y = 0.0, 0.0
        snapped = False
        kind = None
        # Existing nodes snap (exclude the one being dragged if provided)
        if self.point_manager.points:
            for i, p in enumerate(self.point_manager.points):
                if exclude_pid is not None and i == exclude_pid:
                    continue
                dx = x - p["x"]
                dy = y - p["y"]
                if eps_x > 0 and eps_y > 0:
                    if (dx/eps_x)**2 + (dy/eps_y)**2 <= 1.0:
                        x, y = p["x"], p["y"]
                        snapped = True
                        kind = 'node'
                        break
                else:
                    if abs(dx) <= eps_x and abs(dy) <= eps_y:
                        x, y = p["x"], p["y"]
                        snapped = True
                        kind = 'node'
                        break
        # Vertical edges
        if eps_x > 0:
            if abs(x - self.xmin) <= eps_x:
                x = self.xmin
                snapped = True
                kind = 'left'
            elif abs(x - self.xmax) <= eps_x:
                x = self.xmax
                snapped = True
                kind = 'right'
        # Bottom edge
        if eps_y > 0 and abs(y - self.ymin) <= eps_y:
            y = self.ymin
            snapped = True
            kind = 'bottom'
        # Topography
        ytop = self._topo_y(x)
        if ytop is not None and eps_y > 0 and abs(y - ytop) <= eps_y:
            y = ytop
            snapped = True
            kind = 'topo'
        return x, y, snapped, kind

    def _clear_snap_hint(self):
        for a in self._snap_hint_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._snap_hint_artists = []
        self.canvas.draw_idle()

    def _show_snap_hint(self, kind, x, y):
        # Remove any existing hint
        self._clear_snap_hint()
        # Choose label and color
        label = {
            'topo': 'snap: topo',
            'left': 'snap: left edge',
            'right': 'snap: right edge',
            'bottom': 'snap: bottom edge',
            'node': 'snap: node',
        }.get(kind, 'snap')
        color = '#1f77b4' if kind == 'topo' else ('#2ca02c' if kind == 'node' else '#d62728')
        try:
            dot, = self.ax.plot([x], [y], marker='o', color=color, markersize=6, zorder=7)
            txt = self.ax.text(x, y, f"  {label}", color=color, fontsize=9, zorder=7,
                               va='center', ha='left', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))
            self._snap_hint_artists.extend([dot, txt])
            self.canvas.draw_idle()
            # Auto-clear after a short delay
            QTimer.singleShot(700, self._clear_snap_hint)
        except Exception:
            pass

    def _restore_highlights(self):
        if not self.node_edit_active:
            return
        if self._active_highlight_pid is None:
            return
        # Redraw highlight overlays for the active pid
        self._highlight_point_lines(self._active_highlight_pid)

    def _delete_point(self, pid):
        """Delete a point and reconnect lines, unless it's an original topo point."""
        if pid < 0 or pid >= len(self.point_manager.points):
            return
        p = self.point_manager.points[pid]
        # Prevent deletion of original topography points
        if p.get("topo_original", False):
            self.statusBar().showMessage("Cannot delete original topography point", 2000)
            return
        # Get connected lines
        connected_lines = p.get("lines", [])
        if len(connected_lines) == 0:
            # Orphan point, just remove
            del self.point_manager.points[pid]
            self.point_manager.npoint -= 1
            self._renumber_points_after_delete(pid)
            self.plot_model()
            self._selected_pid = None
            self._active_highlight_pid = None
            return
        if len(connected_lines) == 2:
            # Typical case: merge two lines into one
            l1_idx = connected_lines[0]
            l2_idx = connected_lines[1]
            l1 = self.line_manager.lines[l1_idx]
            l2 = self.line_manager.lines[l2_idx]
            # Find the other endpoints
            other1 = l1["point1"] if l1["point2"] == pid else l1["point2"]
            other2 = l2["point1"] if l2["point2"] == pid else l2["point2"]
            # Create new line connecting other1 and other2
            bodies = list(set(l1.get("bodies", []) + l2.get("bodies", [])))
            self.line_manager.append_line(other1, other2, body=bodies)
            new_line_idx = self.line_manager.nline
            # Remove old lines from all point references first
            for pt in self.point_manager.points:
                if l1_idx in pt.get("lines", []):
                    pt["lines"].remove(l1_idx)
                if l2_idx in pt.get("lines", []):
                    pt["lines"].remove(l2_idx)
            
            # Update point-line references with new line
            self.point_manager.points[other1]["lines"].append(new_line_idx)
            self.point_manager.points[other2]["lines"].append(new_line_idx)
            # Update bodies: replace old lines with new
            for body in self.body_manager.bodies:
                for i in range(len(body["lines"])):
                    if body["lines"][i] == l1_idx:
                        body["lines"][i] = new_line_idx
                        # Keep same sense
                    elif body["lines"][i] == l2_idx:
                        # Remove this entry by marking
                        body["lines"][i] = -999
                body["lines"] = [l for l in body["lines"] if l != -999]
                body["sense"] = body["sense"][:len(body["lines"])]
            # Mark old lines as deleted (set to None or use a flag)
            self.line_manager.lines[l1_idx] = None
            self.line_manager.lines[l2_idx] = None
        else:
            # Complex case or endpoint: just disconnect
            for l_idx in connected_lines:
                if l_idx < len(self.line_manager.lines) and self.line_manager.lines[l_idx] is not None:
                    self.line_manager.lines[l_idx] = None
        # Clean up None lines and compact the list BEFORE deleting the point
        self._compact_lines()
        
        # Remove point
        del self.point_manager.points[pid]
        self.point_manager.npoint -= 1
        self._renumber_points_after_delete(pid)
        
        # Clear selection only if deleted point was selected
        if self._selected_pid == pid:
            self._selected_pid = None
        if self._active_highlight_pid == pid:
            self._active_highlight_pid = None
        self.plot_model()
        self.statusBar().showMessage("Point deleted", 2000)
    def _project_to_domain_border(self, x0, y0, x1, y1):
        """Project the segment from (x0,y0) to (x1,y1) onto the model domain border.
        Returns (xb, yb, True) if intersection with [xmin,xmax]x[ymin,ymax] exists, else (x1,y1, False)."""
        try:
            xmin, xmax = float(self.xmin), float(self.xmax)
            ymin, ymax = float(self.ymin), float(self.ymax)
        except Exception:
            return x1, y1, False
        dx = x1 - x0
        dy = y1 - y0
        eps = 1e-12
        candidates = []
        if abs(dx) > eps:
            t = (xmin - x0) / dx
            y = y0 + t * dy
            if t >= 0.0 and t <= 1.0 and y >= ymin - 1e-9 and y <= ymax + 1e-9:
                candidates.append((t, xmin, y))
            t = (xmax - x0) / dx
            y = y0 + t * dy
            if t >= 0.0 and t <= 1.0 and y >= ymin - 1e-9 and y <= ymax + 1e-9:
                candidates.append((t, xmax, y))
        if abs(dy) > eps:
            t = (ymin - y0) / dy
            x = x0 + t * dx
            if t >= 0.0 and t <= 1.0 and x >= xmin - 1e-9 and x <= xmax + 1e-9:
                candidates.append((t, x, ymin))
            t = (ymax - y0) / dy
            x = x0 + t * dx
            if t >= 0.0 and t <= 1.0 and x >= xmin - 1e-9 and x <= xmax + 1e-9:
                candidates.append((t, x, ymax))
        if not candidates:
            return x1, y1, False
        candidates = [c for c in candidates if c[0] > 0.0]
        if not candidates:
            return x1, y1, False
        t_min, xb, yb = sorted(candidates, key=lambda c: c[0])[0]
        return xb, yb, True

    def _clamp_to_domain(self, x, y):
        """Clamp a point to the model domain. Top border follows topography if available.
        Returns a point on or inside the domain rectangle."""
        try:
            xmin, xmax = float(self.xmin), float(self.xmax)
            ymin, ymax = float(self.ymin), float(self.ymax)
        except Exception:
            return x, y
        # Clamp x first to domain range
        xc = min(max(x, xmin), xmax)
        # Determine top boundary at xc (topography or ymax)
        yt = self._topo_y(xc)
        ytop = yt if yt is not None else ymax
        # Clamp y to [ymin, ytop]
        yc = min(max(y, ymin), ytop)
        return xc, yc

    def _renumber_points_after_delete(self, deleted_pid):
        """Adjust point indices in lines and bodies after deleting a point."""
        # Decrement point indices > deleted_pid in all lines
        for line in self.line_manager.lines:
            if line is None:
                continue
            if line["point1"] > deleted_pid:
                line["point1"] -= 1
            if line["point2"] > deleted_pid:
                line["point2"] -= 1

    # --- Body editing event handlers ---
    def _on_body_key_press(self, event):
        if not self.body_edit_active:
            return
        if event.key in ('enter', 'return'):
            self._save_model_if_configured()
            self.stop_body_edit_mode()
        elif event.key == 'escape':
            # Restore backup
            if isinstance(self._body_edit_backup, dict):
                try:
                    self.point_manager.points = copy.deepcopy(self._body_edit_backup.get("points", []))
                    self.line_manager.lines = copy.deepcopy(self._body_edit_backup.get("lines", []))
                    self.body_manager.bodies = copy.deepcopy(self._body_edit_backup.get("bodies", []))
                    self.point_manager.npoint = len(self.point_manager.points) - 1
                    self.line_manager.nline = len(self.line_manager.lines) - 1
                    self.body_manager.nbody = len(self.body_manager.bodies) - 1
                    self._refresh_screen_coords()
                    self.plot_model()
                except Exception:
                    pass
            self.stop_body_edit_mode()

    def _on_body_mouse_press(self, event):
        if not self.body_edit_active:
            return
        
        # Check if we're in join mode waiting for second body (accept any button)
        if self._body_join_first is not None:
            if event.xdata is None or event.ydata is None:
                return
            bid, _ = self.body_manager.inside_body(event.xdata, event.ydata, self.point_manager.points, self.line_manager.lines)
            if bid < 0:
                self.statusBar().showMessage("No body at click location", 2000)
                return
            if bid != self._body_join_first:
                # Second selection - join
                self._join_bodies(self._body_join_first, bid)
                self._body_join_first = None
            else:
                self.statusBar().showMessage("Cannot join body with itself", 2000)
                self._body_join_first = None
            return
        
        if event.button == 1:
            # Start split line drag; accept clicks outside by snapping/clamping to border
            x0 = event.xdata
            y0 = event.ydata
            if x0 is None or y0 is None:
                # Map screen to data and clamp to domain
                try:
                    xd, yd = self.ax.transData.inverted().transform((event.x, event.y))
                except Exception:
                    return
                # Try snapping to edges/topo if within threshold ellipse
                xs, ys, snapped, _ = self._apply_snapping(xd, yd, exclude_pid=None)
                if snapped:
                    x0, y0 = xs, ys
                else:
                    # Clamp to domain border (top border via topo if available)
                    x0, y0 = self._clamp_to_domain(xd, yd)
            self._body_split_start = (x0, y0)
            self._body_split_last = (x0, y0)

    def _on_body_mouse_move(self, event):
        if not self.body_edit_active:
            return
        if self._body_split_start is not None:
            # Draw preview line to current point or projected border if outside
            if self._body_split_line is not None:
                try:
                    self._body_split_line.remove()
                except Exception:
                    pass
            x0, y0 = self._body_split_start
            if event.xdata is not None and event.ydata is not None:
                x1, y1 = event.xdata, event.ydata
                self._body_split_last = (x1, y1)
            else:
                # Outside axes: project cursor to domain border in data coords
                try:
                    x_try, y_try = self.ax.transData.inverted().transform((event.x, event.y))
                    x1, y1, ok = self._project_to_domain_border(x0, y0, x_try, y_try)
                    if not ok and self._body_split_last is not None:
                        x1, y1 = self._body_split_last
                except Exception:
                    if self._body_split_last is not None:
                        x1, y1 = self._body_split_last
                    else:
                        return
            self._body_split_line, = self.ax.plot([x0, x1], [y0, y1], 'r--', linewidth=2, zorder=6)
            self.canvas.draw_idle()

    def _on_body_mouse_release(self, event):
        if not self.body_edit_active:
            return
        if event.button == 1 and self._body_split_start is not None:
            # Execute split
            x0, y0 = self._body_split_start
            if event.xdata is not None and event.ydata is not None:
                x1, y1 = event.xdata, event.ydata
            else:
                # Try to project to domain border using screen coords
                try:
                    x_try, y_try = self.ax.transData.inverted().transform((event.x, event.y))
                    x1, y1, ok = self._project_to_domain_border(x0, y0, x_try, y_try)
                    if not ok:
                        # Fallback to last in-axes point if available
                        if self._body_split_last is not None:
                            x1, y1 = self._body_split_last
                        else:
                            # Abort if we cannot determine an endpoint
                            self._body_split_start = None
                            if self._body_split_line is not None:
                                try:
                                    self._body_split_line.remove()
                                except Exception:
                                    pass
                                self._body_split_line = None
                            self.canvas.draw_idle()
                            return
                except Exception:
                    if self._body_split_last is not None:
                        x1, y1 = self._body_split_last
                    else:
                        self._body_split_start = None
                        if self._body_split_line is not None:
                            try:
                                self._body_split_line.remove()
                            except Exception:
                                pass
                            self._body_split_line = None
                        self.canvas.draw_idle()
                        return
            self._split_bodies_with_line(x0, y0, x1, y1)
            self._body_split_start = None
            self._body_split_last = None
            if self._body_split_line is not None:
                try:
                    self._body_split_line.remove()
                except Exception:
                    pass
                self._body_split_line = None
            self.plot_model()
        elif event.button == 3:
            # Right-click: select first body for joining
            if event.xdata is None or event.ydata is None:
                return
            bid, _ = self.body_manager.inside_body(event.xdata, event.ydata, self.point_manager.points, self.line_manager.lines)
            if bid < 0:
                self.statusBar().showMessage("No body at click location", 2000)
                return
            # First selection
            self._body_join_first = bid
            self.statusBar().showMessage(f"Body {bid} selected. Click adjacent body to join.", 3000)

    def _split_bodies_with_line(self, x0, y0, x1, y1):
        """Split all bodies crossed by line (x0,y0)-(x1,y1)."""
        # Keep splitting until no more bodies can be split
        # Re-detect crossings after each successful split since geometry changes
        total_splits = 0
        max_iterations = 50  # Safety limit
        failed = set()
        split_lines = set()  # Track line indices created by splits to exclude from crossing detection

        for iteration in range(max_iterations):
            # Find all bodies crossed by this line (skip those that already failed)
            crossed_bodies = []
            for ib, _ in enumerate(self.body_manager.bodies):
                if ib in failed:
                    continue
                crossings = self._find_body_crossings(ib, x0, y0, x1, y1, exclude_lines=split_lines)
                if len(crossings) >= 2:
                    crossed_bodies.append((ib, crossings))
                    # crossing count debug removed

            if not crossed_bodies:
                break

            progressed = False
            # Try to split each crossed body; if one succeeds, restart detection
            for ib, crossings in crossed_bodies:
                # iteration split debug removed
                ok, new_split_line = self._split_body_at_crossings(ib, crossings, x0, y0, x1, y1)
                if ok:
                    total_splits += 1
                    progressed = True
                    if new_split_line is not None:
                        split_lines.add(new_split_line)
                    break  # Re-detect after geometry change
                else:
                    failed.add(ib)

            if not progressed:
                # No further valid splits possible this round
                break

        if total_splits == 0:
            self.statusBar().showMessage(f"Line does not completely cross any body (checked {len(self.body_manager.bodies)} bodies)", 3000)
        else:
            self.statusBar().showMessage(f"Split {total_splits} body/bodies", 2000)
            # total splits debug removed

    def _find_body_crossings(self, body_idx, x0, y0, x1, y1, exclude_lines=None):
        """Find exactly TWO intersection points where split line crosses body boundary.
        Returns list of exactly 0 or 2 crossing tuples: (line_idx, t_param, x_cross, y_cross, point_idx_or_None).
        If the line doesn't properly cross the body, returns empty list.
        
        Args:
            exclude_lines: Set of line indices to exclude from crossing detection (e.g., split lines)
        """
        if exclude_lines is None:
            exclude_lines = set()
        body = self.body_manager.bodies[body_idx]
        raw = []
        
        # Find all potential edge intersections
        for i, lin_idx in enumerate(body["lines"]):
            # Skip split lines created during this operation
            if lin_idx in exclude_lines:
                print(f"DEBUG: Skipping split line {lin_idx}")
                continue
            line = self.line_manager.lines[lin_idx]
            if line is None:
                continue
            p1 = self.point_manager.points[line["point1"]]
            p2 = self.point_manager.points[line["point2"]]
            
            # Compute segment intersection
            xi, yi, t_split, t_edge, intersects = self._segment_intersection(
                x0, y0, x1, y1, p1["x"], p1["y"], p2["x"], p2["y"]
            )
            if not intersects or t_edge is None or t_split is None:
                continue
            
            # Only accept intersections strictly INSIDE the edge (not at endpoints)
            # This avoids double-counting at vertices where two edges meet
            edge_tol = 1e-5
            if edge_tol < t_edge < (1.0 - edge_tol):
                # This is a clean interior crossing
                pid_snap = self._find_point_within_ellipse(xi, yi)
                print(f"DEBUG: Found crossing at ({xi:.2f},{yi:.2f}) t_split={t_split:.4f} t_edge={t_edge:.4f} on edge {lin_idx} ({p1['x']:.2f},{p1['y']:.2f})->({p2['x']:.2f},{p2['y']:.2f})")
                raw.append((lin_idx, float(t_split), float(xi), float(yi), pid_snap))
        
        # If we found 0 or 1 crossings, the line doesn't properly cross the body
        if len(raw) < 2:
            print(f"DEBUG: Only found {len(raw)} crossings, need 2")
            return []
        
        # Sort by parameter along the split line
        raw.sort(key=lambda c: c[1])
        
        print(f"DEBUG: Total raw crossings: {len(raw)}, keeping first and last")
        
        # Return ONLY the first and last crossing (entry and exit points)
        # Any intermediate crossings are artifacts and should be ignored
        return [raw[0], raw[-1]]

    def _segment_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Find intersection of segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4).
        Returns (xi, yi, t1, t2, intersects) where t1,t2 strictly in (0,1) or at endpoints.
        Only returns True if intersection is genuinely on BOTH segments (not extrapolated)."""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        eps = 1e-10
        if abs(denom) < eps:
            return None, None, None, None, False
        t1 = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        t2 = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        # Strict check: both parameters must be in [0,1] with no extrapolation
        if -eps <= t1 <= 1+eps and -eps <= t2 <= 1+eps:
            # Clamp to [0,1] to avoid numerical noise
            t1 = max(0.0, min(1.0, t1))
            t2 = max(0.0, min(1.0, t2))
            xi = x1 + t1 * (x2 - x1)
            yi = y1 + t1 * (y2 - y1)
            return xi, yi, t1, t2, True
        return None, None, None, None, False

    def _find_point_within_ellipse(self, x, y):
        """Return point index if (x,y) is within eps ellipse of existing point, else None."""
        try:
            eps_x = float(getattr(self.point_manager, 'eps_x', 0.0) or 0.0)
            eps_y = float(getattr(self.point_manager, 'eps_y', 0.0) or 0.0)
        except Exception:
            return None
        if eps_x <= 0 or eps_y <= 0:
            return None
        for i, p in enumerate(self.point_manager.points):
            dx = (x - p["x"]) / eps_x
            dy = (y - p["y"]) / eps_y
            if dx*dx + dy*dy <= 1.0:
                return i
        return None

    def _split_body_at_crossings(self, body_idx, crossings, x0, y0, x1, y1):
        """Split a body using exactly TWO crossing points. Returns (success, split_line_idx) tuple."""
        # Must have exactly 2 crossings (entry and exit)
        if len(crossings) != 2:
            if len(crossings) > 2:
                self.statusBar().showMessage(f"Warning: {len(crossings)} crossings detected, expected 2", 2000)
            return False, None
        
        first_crossing = crossings[0]
        last_crossing = crossings[1]
        
        # Record which body line each crossing is on BEFORE insertion
        body = self.body_manager.bodies[body_idx]
        first_body_line_idx = body["lines"].index(first_crossing[0]) if first_crossing[0] in body["lines"] else 0
        last_body_line_idx = body["lines"].index(last_crossing[0]) if last_crossing[0] in body["lines"] else len(body["lines"]) - 1
        # body line indices debug removed
        
        # Insert points at crossings (or use existing)
        crossing_pids = []
        for crossing in [first_crossing, last_crossing]:
            lin_idx, t, xi, yi, pid_snap = crossing
            if pid_snap is not None:
                crossing_pids.append(pid_snap)
            else:
                # Insert new point on this line
                xs, ys = self.ax.transData.transform((xi, yi))
                pnew = self._split_line_with_point(lin_idx, xi, yi, xs, ys)
                crossing_pids.append(pnew)
        
        # Compact lines after insertions
        self._compact_lines()
        
        # Re-find crossing positions in body's line list after compaction
        body = self.body_manager.bodies[body_idx]
        crossing_line_indices = []
        for pid in crossing_pids:
            p = self.point_manager.points[pid]
            # Find first line in body that contains this point
            for line_idx in p["lines"]:
                if line_idx in body["lines"]:
                    idx = body["lines"].index(line_idx)
                    crossing_line_indices.append(idx)
                    break
        
        if len(crossing_line_indices) < 2:
            self.statusBar().showMessage("Failed to locate crossing points in body", 2000)
            return False, None
        
        # Create new split line connecting the first and last crossing points
        p_start = crossing_pids[0]
        p_end = crossing_pids[1]
        self.line_manager.append_line(p_start, p_end, body=[])
        split_line_idx = self.line_manager.nline
        # Update point-line references
        self.point_manager.points[p_start]["lines"].append(split_line_idx)
        self.point_manager.points[p_end]["lines"].append(split_line_idx)
        
        # Build ordered boundary points for this body using its current lines and senses
        # This creates a cyclic vertex list [v0, v1, ..., vn] where consecutive pairs are edges
        ordered_points = []
        if not body["lines"]:
            self.statusBar().showMessage("Body has no lines to split", 2000)
            return False, None
        # Initialize with first line and sense
        first_line_idx = body["lines"][0]
        first_sense = body["sense"][0]
        first_line = self.line_manager.lines[first_line_idx]
        if first_sense >= 0:
            ordered_points = [first_line["point1"], first_line["point2"]]
        else:
            ordered_points = [first_line["point2"], first_line["point1"]]
        # Walk remaining lines to build point sequence
        for i in range(1, len(body["lines"])):
            lidx = body["lines"][i]
            sgn = body["sense"][i]
            lobj = self.line_manager.lines[lidx]
            last_pt = ordered_points[-1]
            if sgn >= 0:
                # Expect lobj.point1 == last_pt
                if lobj["point1"] != last_pt:
                    # Fallback: try reversed
                    if lobj["point2"] == last_pt:
                        ordered_points.append(lobj["point1"])  # reversed
                    else:
                        # Topology issue; abort
                        # topology mismatch debug removed
                        self.statusBar().showMessage("Failed to build body boundary order", 2000)
                        return False, None
                else:
                    ordered_points.append(lobj["point2"])
            else:
                # Expect lobj.point2 == last_pt
                if lobj["point2"] != last_pt:
                    if lobj["point1"] == last_pt:
                        ordered_points.append(lobj["point2"])  # reversed
                    else:
                        # topology mismatch debug removed
                        self.statusBar().showMessage("Failed to build body boundary order", 2000)
                        return False, None
                else:
                    ordered_points.append(lobj["point1"])

        # Remove immediate duplicate vertices that can arise from degenerate sequences
        def dedup_consecutive(seq):
            if not seq:
                return seq
            out = [seq[0]]
            for v in seq[1:]:
                if v != out[-1]:
                    out.append(v)
            return out
        ordered_points = dedup_consecutive(ordered_points)

        # Locate crossing points within ordered_points
        p_start = crossing_pids[0]
        p_end = crossing_pids[1]
        try:
            i1 = ordered_points.index(p_start)
            i2 = ordered_points.index(p_end)
        except ValueError:
            self.statusBar().showMessage("Crossing points not found on body boundary", 2000)
            return False, None
        npts = len(ordered_points)
        # partitioning vertices debug removed

        # Degeneracy guard: if crossings are the same or adjacent vertices, the split
        # just touches a corner and does not produce two valid polygons.
        step = (i2 - i1) % npts
        if step in (0, 1, npts - 1):
            self.statusBar().showMessage("Split touches boundary corner; skipping this body", 3000)
            return False, None

        # Helper to iterate circularly from a to b inclusive (forward)
        def forward_path(a, b):
            i = a
            path = [ordered_points[i]]
            while i != b:
                i = (i + 1) % npts
                path.append(ordered_points[i])
            return path

        # Construct the two vertex paths
        verts1 = forward_path(i1, i2)  # path from p_start to p_end
        verts2 = forward_path(i2, i1)  # complementary path from p_end to p_start

        # Map consecutive vertex pairs to line indices and senses
        def lines_from_verts(verts):
            ls = []
            ss = []
            for a, b in zip(verts[:-1], verts[1:]):
                # Skip degenerate zero-length edges
                if a == b:
                    continue
                found = False
                for il, l in enumerate(self.line_manager.lines):
                    if l is None:
                        continue
                    # Only use edges that belong to this body
                    if il not in body["lines"]:
                        continue
                    if l["point1"] == a and l["point2"] == b:
                        ls.append(il); ss.append(+1); found = True; break
                    if l["point1"] == b and l["point2"] == a:
                        ls.append(il); ss.append(-1); found = True; break
                if not found:
                    # missing edge debug removed
                    return [], []
            return ls, ss

        body1_lines, body1_sense = lines_from_verts(verts1)
        body2_lines, body2_sense = lines_from_verts(verts2)

        if not body1_lines or not body2_lines:
            self.statusBar().showMessage("Failed to derive edges from vertex paths", 2000)
            return False, None

        # Close polygons by adding the split line (ensure direction matches)
        # split_line is from p_start -> p_end
        # For body1 (verts1: p_start..p_end), we need split edge p_end -> p_start, i.e., sense -1
        body1_lines.append(split_line_idx)
        body1_sense.append(-1)
        # For body2 (verts2: p_end..p_start), we need split edge p_start -> p_end, i.e., sense +1
        body2_lines.append(split_line_idx)
        body2_sense.append(+1)
        
        # parts length debug removed
        
        # Validate we have different sets
        # Additional guard: reject splits producing polygons with <3 boundary edges
        if len(body1_lines) < 3 or len(body2_lines) < 3:
            self.statusBar().showMessage(f"Split created invalid bodies: part1={len(body1_lines)} part2={len(body2_lines)}", 3000)
            # invalid body sizes debug removed
            return False, None
        
        # creating bodies debug removed
        
        # Update original body with part 1
        # updating body debug removed
        body["lines"] = body1_lines
        body["sense"] = body1_sense
        
        # Create new body with part 2
        new_body_idx = self.body_manager.nbody + 1
        # creating new body debug removed
        self.body_manager.append_body(
            body2_lines, body2_sense,
            body["prop_names"], para_values=list(body["props"]),
            name=body["name"] + "_split"
        )
        # body manager count debug removed
        
        # Update line body references
        self.line_manager.lines[split_line_idx]["bodies"] = [body_idx, new_body_idx]
        for l_idx in body1_lines:
            if l_idx != split_line_idx:
                line = self.line_manager.lines[l_idx]
                if body_idx not in line.get("bodies", []):
                    line["bodies"].append(body_idx)
        for l_idx in body2_lines:
            if l_idx != split_line_idx:
                line = self.line_manager.lines[l_idx]
                if new_body_idx not in line.get("bodies", []):
                    line["bodies"].append(new_body_idx)
        
        self.statusBar().showMessage(f"Body {body_idx} split into bodies {body_idx} and {new_body_idx}", 3000)
        return True, split_line_idx

    def _join_bodies(self, body1_idx, body2_idx):
        """Join two adjacent bodies if they share a common line."""
        body1 = self.body_manager.bodies[body1_idx]
        body2 = self.body_manager.bodies[body2_idx]
        # Find common lines (shared edges)
        common_line_set = set(body1["lines"]) & set(body2["lines"])
        if not common_line_set:
            self.statusBar().showMessage("Bodies do not share a common line", 2000)
            return
        # Build new boundary by walking both polygons and removing shared edges
        # Strategy: traverse body1, then body2, skipping common lines
        new_lines = []
        new_sense = []
        # Add body1 lines except common ones
        for i, l_idx in enumerate(body1["lines"]):
            if l_idx not in common_line_set:
                new_lines.append(l_idx)
                new_sense.append(body1["sense"][i])
        # Find insertion point: last endpoint of body1's boundary before common edge
        # For simplicity, append body2's non-common edges
        # But we need correct orientation: walk body2's boundary excluding common lines
        # and potentially reverse if needed
        # Simplified: add body2 non-common lines with their original sense
        # (This may need topology refinement for complex cases)
        for i, l_idx in enumerate(body2["lines"]):
            if l_idx not in common_line_set:
                new_lines.append(l_idx)
                new_sense.append(body2["sense"][i])
        # Validate and reorder to form a closed polygon
        # Walk the edges to ensure continuity
        ordered_lines, ordered_sense = self._reorder_polygon_edges(new_lines, new_sense)
        if not ordered_lines:
            self.statusBar().showMessage("Failed to merge body boundaries correctly", 2000)
            return
        # Update body1 with merged boundary
        body1["lines"] = ordered_lines
        body1["sense"] = ordered_sense
        
        # Remove common lines from point references BEFORE deleting the lines
        for l_idx in common_line_set:
            line = self.line_manager.lines[l_idx]
            # Remove this line from both endpoints
            p1 = self.point_manager.points[line["point1"]]
            p2 = self.point_manager.points[line["point2"]]
            if l_idx in p1.get("lines", []):
                p1["lines"].remove(l_idx)
            if l_idx in p2.get("lines", []):
                p2["lines"].remove(l_idx)
        
        # Update line body references and mark common lines for deletion
        for l_idx in common_line_set:
            line = self.line_manager.lines[l_idx]
            if body1_idx in line["bodies"]:
                line["bodies"].remove(body1_idx)
            if body2_idx in line["bodies"]:
                line["bodies"].remove(body2_idx)
            # Mark line as deleted
            self.line_manager.lines[l_idx] = None
        for l_idx in ordered_lines:
            line = self.line_manager.lines[l_idx]
            if body2_idx in line["bodies"]:
                line["bodies"].remove(body2_idx)
            if body1_idx not in line["bodies"]:
                line["bodies"].append(body1_idx)
        # Remove body2
        del self.body_manager.bodies[body2_idx]
        self.body_manager.nbody -= 1
        # Renumber body references in lines
        for line in self.line_manager.lines:
            if line is None:
                continue
            line["bodies"] = [b if b < body2_idx else b - 1 for b in line["bodies"]]
        
        # Compact lines to remove deleted common edges
        self._compact_lines()
        
        # Clean up orphaned points (points not connected to any lines)
        self._remove_orphaned_points()
        
        # Refresh the display
        self.plot_model()
        
        self.statusBar().showMessage(f"Joined bodies {body1_idx} and {body2_idx}", 2000)

    def _reorder_polygon_edges(self, lines, senses):
        """Reorder edges to form a continuous closed polygon."""
        if not lines:
            return [], []
        # Build adjacency: for each endpoint, track which lines connect to it
        # and walk to form a closed loop
        ordered = []
        ordered_sense = []
        used = [False] * len(lines)
        # Start with first line
        current_line = lines[0]
        current_sense = senses[0]
        ordered.append(current_line)
        ordered_sense.append(current_sense)
        used[0] = True
        # Get current endpoint
        line_obj = self.line_manager.lines[current_line]
        if current_sense > 0:
            current_end = line_obj["point2"]
        else:
            current_end = line_obj["point1"]
        # Walk until we return to start or use all lines
        start_point = line_obj["point1"] if current_sense > 0 else line_obj["point2"]
        for _ in range(len(lines) - 1):
            # Find next line connected to current_end
            found = False
            for i, l_idx in enumerate(lines):
                if used[i]:
                    continue
                line_obj = self.line_manager.lines[l_idx]
                sense = senses[i]
                if sense > 0:
                    if line_obj["point1"] == current_end:
                        ordered.append(l_idx)
                        ordered_sense.append(sense)
                        used[i] = True
                        current_end = line_obj["point2"]
                        found = True
                        break
                else:
                    if line_obj["point2"] == current_end:
                        ordered.append(l_idx)
                        ordered_sense.append(sense)
                        used[i] = True
                        current_end = line_obj["point1"]
                        found = True
                        break
            if not found:
                # Try reverse matching
                for i, l_idx in enumerate(lines):
                    if used[i]:
                        continue
                    line_obj = self.line_manager.lines[l_idx]
                    sense = senses[i]
                    # Try flipping sense
                    if sense > 0:
                        if line_obj["point2"] == current_end:
                            ordered.append(l_idx)
                            ordered_sense.append(-sense)
                            used[i] = True
                            current_end = line_obj["point1"]
                            found = True
                            break
                    else:
                        if line_obj["point1"] == current_end:
                            ordered.append(l_idx)
                            ordered_sense.append(-sense)
                            used[i] = True
                            current_end = line_obj["point2"]
                            found = True
                            break
            if not found:
                # Cannot continue - topology error
                return [], []
        # Check closure
        if current_end != start_point:
            # Not closed - may need adjustment
            pass
        return ordered, ordered_sense

    def _compact_lines(self):
        """Remove None entries from line list and renumber all references."""
        # Build mapping from old index to new index
        old_to_new = {}
        new_lines = []
        new_idx = 0
        for old_idx, line in enumerate(self.line_manager.lines):
            if line is not None:
                old_to_new[old_idx] = new_idx
                new_lines.append(line)
                new_idx += 1
            else:
                old_to_new[old_idx] = -1  # Deleted
        # Replace line list
        self.line_manager.lines = new_lines
        self.line_manager.nline = len(new_lines) - 1
        # Update all point line references
        for p in self.point_manager.points:
            p["lines"] = [old_to_new[l] for l in p.get("lines", []) if l in old_to_new and old_to_new[l] != -1]
        # Update all body line references
        for body in self.body_manager.bodies:
            new_body_lines = []
            new_body_sense = []
            for i, l in enumerate(body["lines"]):
                if l in old_to_new and old_to_new[l] != -1:
                    new_body_lines.append(old_to_new[l])
                    if i < len(body["sense"]):
                        new_body_sense.append(body["sense"][i])
            body["lines"] = new_body_lines
            body["sense"] = new_body_sense

    def _remove_orphaned_points(self):
        """Remove points that are not connected to any lines."""
        # First, update point line references to remove deleted lines
        for point in self.point_manager.points:
            valid_lines = []
            for l_idx in point.get("lines", []):
                if l_idx < len(self.line_manager.lines) and self.line_manager.lines[l_idx] is not None:
                    valid_lines.append(l_idx)
            point["lines"] = valid_lines
        
        # Find all points that are referenced by at least one line
        used_points = set()
        for line in self.line_manager.lines:
            if line is not None:
                used_points.add(line["point1"])
                used_points.add(line["point2"])
        
        # Find orphaned points
        orphaned = []
        for i, point in enumerate(self.point_manager.points):
            if i not in used_points:
                orphaned.append(i)
        
        if not orphaned:
            return
        
        # Build mapping from old index to new index
        old_to_new = {}
        new_points = []
        new_idx = 0
        for old_idx, point in enumerate(self.point_manager.points):
            if old_idx not in orphaned:
                old_to_new[old_idx] = new_idx
                new_points.append(point)
                new_idx += 1
            else:
                old_to_new[old_idx] = -1  # Deleted
        
        # Replace point list
        self.point_manager.points = new_points
        self.point_manager.npoint = len(new_points) - 1
        
        # Update all line point references
        for line in self.line_manager.lines:
            if line is not None:
                line["point1"] = old_to_new[line["point1"]]
                line["point2"] = old_to_new[line["point2"]]
        
        self.statusBar().showMessage(f"Removed {len(orphaned)} orphaned point(s)", 2000)

    def _clear_highlights(self):
        for artist in self._edit_highlights:
            try:
                artist.remove()
            except Exception:
                pass
        self._edit_highlights = []
        self.canvas.draw_idle()

    def _highlight_point_lines(self, pid):
        # Clear previous highlights
        self._clear_highlights()
        # Draw connected lines in red on top
        for il in self.point_manager.points[pid].get("lines", []):
            if il < 0 or il >= len(self.line_manager.lines):
                continue
            line = self.line_manager.lines[il]
            p1 = self.point_manager.points[line["point1"]]
            p2 = self.point_manager.points[line["point2"]]
            ln, = self.ax.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]], color='red', linewidth=2.0, zorder=5)
            self._edit_highlights.append(ln)
        self.canvas.draw_idle()

    # --- Picks reduction (pickf) ---
    def select_pickf(self):
        """Prompt for receiver/shot decimation based on base picks file, then apply reduction."""
        # Always derive counts from base picks on disk (prefer picks.dat, fallback to picks.sgt)
        base_scheme = None
        base_file = None
        try:
            import pygimli.physics.traveltime as tt
            if os.path.exists("picks.dat"):
                base_scheme = tt.load("picks.dat", verbose=True)
                base_file = "picks.dat"
            elif os.path.exists("picks.sgt"):
                base_scheme = tt.load("picks.sgt", verbose=True)
                base_file = "picks.sgt"
        except Exception:
            base_scheme = None
        if base_scheme is None:
            try:
                QMessageBox.information(self, "Select picks", "No base picks found (picks.dat or picks.sgt).")
            except Exception:
                pass
            return
        try:
            n_rec = int(len(np.unique(base_scheme["g"])))
            n_shot = int(len(np.unique(base_scheme["s"])))
        except Exception:
            n_rec, n_shot = 0, 0
        dlg = SelectPickfDialog(n_rec, n_shot, parent=self)
        if dlg.exec_() != dlg.Accepted:
            return
        vals = dlg.values()
        if not vals:
            return
        g0, dg, s0, ds = vals
        # Validate ranges
        if g0 < 1 or s0 < 1 or dg < 1 or ds < 1 or g0 > max(1, n_rec) or s0 > max(1, n_shot):
            try:
                QMessageBox.warning(self, "Select picks", "Invalid selection parameters.")
            except Exception:
                pass
            return
        try:
            self.reduce_geometry(g0=g0, dg=dg, s0=s0, ds=ds)
            # Recompute positions and replot picks in upper canvas
            self.get_pick_positions()
            try:
                self.ax_dat.cla()
            except Exception:
                pass
            self.ax_dat.set_title("Data plot")
            self.ax_dat.set_ylabel("Time [s]")
            try:
                data = self.scheme["t"]
            except Exception:
                data = None
            if data is not None:
                self.plot_picks(self.ax_dat, data, marker="+")
            self.statusBar().showMessage("Reduced picks applied (based on base picks).", 3000)
        except Exception as e:
            try:
                QMessageBox.warning(self, "Select picks", f"Failed to reduce picks: {e}")
            except Exception:
                pass

    def reduce_geometry(self, g0: int, dg: int, s0: int, ds: int):
        """
        Reduce number of receivers and shot points according to decimation params.

        Parameters
        ----------
        g0 : int
            First geophone to use (natural counting, 1-based).
        dg : int
            Take one geophone out of every dg.
        s0 : int
            First shot to use (natural counting, 1-based).
        ds : int
            Take one shot out of every ds.
        """
        # Always start from base picks on disk: picks.dat preferred, fallback picks.sgt
        try:
            import pygimli.physics.traveltime as tt
            if os.path.exists("picks.dat"):
                scheme = tt.load("picks.dat", verbose=True)
            elif os.path.exists("picks.sgt"):
                scheme = tt.load("picks.sgt", verbose=True)
            else:
                raise FileNotFoundError("picks.dat or picks.sgt not found")
        except Exception as e:
            raise RuntimeError(f"Cannot load base picks: {e}")
        # Determine auxiliary columns besides s,g,t,err
        try:
            keys = list(scheme.dataMap().keys())
        except Exception:
            try:
                keys = list(scheme.keys())  # DataContainer dict-like
            except Exception:
                keys = []
        for k in ["s", "g", "t", "err"]:
            try:
                if k in keys:
                    keys.remove(k)
            except Exception:
                pass

        s_id = np.asarray(scheme["s"], dtype=int)
        g_id = np.asarray(scheme["g"], dtype=int)
        t = np.asarray(scheme["t"], dtype=float)
        try:
            err = np.asarray(scheme["err"], dtype=float)
        except Exception:
            err = np.zeros_like(t) + 0.0001

        extra = {}
        for k in keys:
            try:
                extra[k] = scheme[k]
            except Exception:
                extra[k] = []

        pos = np.array(scheme.sensors())
        # Ensure pos has x,y,z columns for file
        if pos.shape[1] < 3:
            zcol = np.zeros((pos.shape[0], 1), dtype=float)
            pos = np.hstack([pos[:, :2], zcol])

        su = np.unique(s_id)
        gu = np.unique(g_id)

        s_new = []
        g_new = []
        t_new = []
        err_new = []
        extra_new = {k: [] for k in keys}

        # Compute keep indices (convert natural counting to 0-based)
        ishots = list(range(s0 - 1, len(su), ds))
        s_keep = set(su[ishots])
        igeo = list(range(g0 - 1, len(gu), dg))
        g_keep = set(gu[igeo])

        for i, s in enumerate(s_id):
            if s in s_keep and g_id[i] in g_keep:
                s_new.append(int(s))
                g_new.append(int(g_id[i]))
                t_new.append(float(t[i]))
                err_new.append(float(err[i]))
                for k in keys:
                    extra_new[k].append(extra.get(k, [None]*len(s_id))[i])

        # Write temporary sgt file
        fname = "dummy.sgt"
        with open(fname, "w") as fo:
            fo.write(f"{len(pos)} # shot/geophone points\n# x y z\n")
            for x, y, z in pos:
                fo.write(f"{x:0.2f} {y:0.2f} {z:0.2f}\n")
            fo.write(f"{len(t_new)} # measurements\n# s g t err")
            for k in keys:
                fo.write(f" {k}")
            fo.write("\n")
            for i in range(len(t_new)):
                # Write natural counting (add 1)
                fo.write(f"{s_new[i]+1} {g_new[i]+1} {t_new[i]:0.5f} {err_new[i]:0.5f}")
                for k in keys:
                    fo.write(f" {extra_new[k][i]}")
                fo.write("\n")

        # Reload into scheme
        try:
            import pygimli.physics.traveltime as tt
            new_scheme = tt.load(fname, verbose=True)
        finally:
            try:
                os.remove(fname)
            except Exception:
                pass
        self.set_scheme(new_scheme)
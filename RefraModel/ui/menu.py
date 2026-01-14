"""
Menu bar creation for the application
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QMenuBar,
    QAction,
    QApplication,
    QFileDialog,
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt5.QtCore import Qt, QTimer
from .dialogs import CustomDialog, InversionParametersDialog, GeometryDialog,\
    BodyRegularizationDialog, ColorScaleDialog
from ..utils.file_io import FileIO
from .. import version


class Menu:
    """Creates the menu bar for the application"""
    
    def __init__(self, parent):
        self.parent = parent
        self.menu_bar = QMenuBar(parent)
        self.create_menus()
    
    def create_menus(self):
        """Create menus and actions"""
        # File menu
        file_menu = self.menu_bar.addMenu("File")

        save_as_action = QAction("Save model as…", self.menu_bar)
        save_as_action.setShortcut("Ctrl+S")
        save_as_action.triggered.connect(self.save_model_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self.menu_bar)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_application)
        file_menu.addAction(exit_action)

        # Edit menu (placeholders for dedicated editors)
        edit_menu = self.menu_bar.addMenu("Edit")

        body_editor_action = QAction("Body Editor", self.menu_bar)
        body_editor_action.setShortcut("B")
        body_editor_action.triggered.connect(self.start_body_edit_mode)
        edit_menu.addAction(body_editor_action)

        node_editor_action = QAction("Node Editor", self.menu_bar)
        node_editor_action.setShortcut("N")
        node_editor_action.triggered.connect(self.start_node_edit_mode)
        edit_menu.addAction(node_editor_action)

        prop_editor_action = QAction("Property Editor", self.menu_bar)
        prop_editor_action.setShortcut("p")
        prop_editor_action.triggered.connect(self.start_property_edit_mode)
        edit_menu.addAction(prop_editor_action)

        # Tools menu
        tools_menu = self.menu_bar.addMenu("Tools")

        pickf_action = QAction("Select picks", self.menu_bar)
        pickf_action.setShortcut("F4")
        pickf_action.triggered.connect(self.select_picks)
        tools_menu.addAction(pickf_action)

        forward_action = QAction("Run Forward Model", self.menu_bar)
        forward_action.setShortcut("F5")
        forward_action.triggered.connect(self.run_forward_model)
        tools_menu.addAction(forward_action)

        tools_menu.addSeparator()

        inversion_params_action = QAction("Inversion Parameters...", self.menu_bar)
        inversion_params_action.setShortcut("F8")
        inversion_params_action.triggered.connect(self.set_inversion_parameters)
        tools_menu.addAction(inversion_params_action)

        inversion_action = QAction("Run Inversion", self.menu_bar)
        inversion_action.setShortcut("F9")
        inversion_action.triggered.connect(self.run_inversion)
        tools_menu.addAction(inversion_action)
        
        tools_menu.addSeparator()
        
        clear_body_reg_action = QAction("Clear Body Regularization Settings", self.menu_bar)
        clear_body_reg_action.triggered.connect(self.clear_body_regularization)
        tools_menu.addAction(clear_body_reg_action)

        # View menu
        view_menu = self.menu_bar.addMenu("View")
        self.toggle_plot_action = QAction("Toggle Model Plot", self.menu_bar)
        self.toggle_plot_action.setShortcut("Ctrl+T")
        self.toggle_plot_action.triggered.connect(self.toggle_plot)
        self.toggle_plot_action.setEnabled(False)  # Initially disabled until inversion
        view_menu.addAction(self.toggle_plot_action)
        
        self.toggle_rays_action = QAction("Toggle Ray Plot", self.menu_bar)
        self.toggle_rays_action.setShortcut("Ctrl+R")
        self.toggle_rays_action.triggered.connect(self.toggle_rays)
        self.toggle_rays_action.setEnabled(False)  # Initially disabled until forward model or inversion
        view_menu.addAction(self.toggle_rays_action)
        
        self.view_mesh_action = QAction("View Mesh", self.menu_bar)
        self.view_mesh_action.setShortcut("Ctrl+M")
        self.view_mesh_action.triggered.connect(self.view_mesh)
        self.view_mesh_action.setEnabled(False)  # Initially disabled until forward model
        view_menu.addAction(self.view_mesh_action)
        
        view_menu.addSeparator()
        
        self.color_scale_action = QAction("Color Scale Settings...", self.menu_bar)
        self.color_scale_action.setShortcut("Ctrl+L")
        self.color_scale_action.triggered.connect(self.configure_color_scale)
        self.color_scale_action.setEnabled(False)  # Initially disabled until inversion
        view_menu.addAction(self.color_scale_action)

        # Help menu
        help_menu = self.menu_bar.addMenu("Help")
        about_action = QAction("About", self.menu_bar)
        about_action.triggered.connect(self.about_dialog)
        help_menu.addAction(about_action)

        help_action = QAction("Help", self.menu_bar)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.help_dialog)
        help_menu.addAction(help_action)
    
    def exit_application(self):
        """Exit the application, but block if editing is active"""
        if hasattr(self.parent, 'editing_in_progress'):
            if self.parent.editing_in_progress():
                try:
                    QMessageBox.warning(self.parent, "Finish Editing", "Please finish or cancel the current editing operation (ENTER or ESC) before exiting.")
                except Exception as e:
                    QMessageBox.warning(None, "Finish Editing", "Please finish or cancel the current editing operation (ENTER or ESC) before exiting.")
                return
        choice = QMessageBox.question(
            self.parent, "Confirm", "Are you sure?", QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.No:
            return
        print("Exiting application.")
        self.parent.close()
        QApplication.quit()

    # --- File actions ---
    def new_model(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._new_model_impl)
        return self._new_model_impl()

    def _new_model_impl(self):
        try:
            # Clear in-memory managers if available
            if hasattr(self.parent, "body_manager"):
                self.parent.body_manager.bodies.clear()
                self.parent.body_manager.nbody = -1
            if hasattr(self.parent, "point_manager"):
                self.parent.point_manager.points.clear()
                self.parent.point_manager.npoint = -1
            if hasattr(self.parent, "line_manager"):
                self.parent.line_manager.lines.clear()
                self.parent.line_manager.nline = -1
            if hasattr(self.parent, "statusBar"):
                self.parent.statusBar().showMessage("New model created", 3000)
            print("New model initialized.")
        except Exception as e:
            QMessageBox.warning(self.parent, "New Model", f"Failed to create new model: {e}")

    def open_model(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._open_model_impl)
        return self._open_model_impl()

    def _open_model_impl(self):
        filename, _ = QFileDialog.getOpenFileName(self.parent, "Open Model", "", "JSON Files (*.json);;All Files (*.*)")
        if not filename:
            return
        try:
            data = FileIO.load_from_file(filename)
            # Placeholder: just report what was loaded
            QMessageBox.information(self.parent, "Open Model", f"Loaded model summary:\n{data}")
        except Exception as e:
            QMessageBox.warning(self.parent, "Open Model", f"Failed to open model: {e}")

    def save_model_as(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._save_model_as_impl)
        return self._save_model_as_impl()

    def _save_model_as_impl(self):
        import os
        current = getattr(self.parent, "model_save_path", None)
        default_name = os.path.basename(current) if current else "model.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Save model as",
            default_name,
            "Model/Text Files (*.txt *.csv *.mod);;All Files (*.*)"
        )
        if not filename:
            return
        try:
            if hasattr(self.parent, "save_model_to_file"):
                self.parent.save_model_to_file(filename)
            # Update future auto-saves to this new path
            self.parent.model_save_path = filename
            if hasattr(self.parent, "statusBar"):
                self.parent.statusBar().showMessage(f"Saved model to {filename}", 3000)
        except Exception as e:
            QMessageBox.warning(self.parent, "Save model as", f"Failed to save model: {e}")

    # --- Edit helpers ---
    def show_info_dialog(self, message):
        dlg = CustomDialog(message)
        dlg.exec_()

    def start_node_edit_mode(self):
        if hasattr(self.parent, "start_node_edit_mode"):
            self.parent.start_node_edit_mode()

    def start_body_edit_mode(self):
        if hasattr(self.parent, "start_body_edit_mode"):
            self.parent.start_body_edit_mode()

    def start_property_edit_mode(self):
        if hasattr(self.parent, "start_property_edit_mode"):
            self.parent.start_property_edit_mode()

    # --- Tools actions ---
    def build_mesh(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._build_mesh_impl)
        return self._build_mesh_impl()

    def _build_mesh_impl(self):
        try:
            geometry = {
                "points": len(getattr(self.parent.point_manager, "points", [])),
                "lines": len(getattr(self.parent.line_manager, "lines", [])),
                "bodies": len(getattr(self.parent.body_manager, "bodies", [])),
            }
            if hasattr(self.parent, "mesh_builder"):
                self.parent.mesh_builder.create_mesh(geometry)
        except Exception as e:
            QMessageBox.warning(self.parent, "Build Mesh", f"Failed to build mesh: {e}")

    def run_forward_model(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._run_forward_model_impl)
        return self._run_forward_model_impl()

    def _run_forward_model_impl(self):
        import os
        import pygimli.physics.traveltime as tt
        try:
            if not hasattr(self.parent, "forward_model"):
                QMessageBox.warning(self.parent, "Forward Model", "Forward model not initialized")
                return
            # Check if we have bodies
            if not hasattr(self.parent, "body_manager") or len(self.parent.body_manager.bodies) == 0:
                QMessageBox.warning(self.parent, "Forward Model", "No bodies defined in the model")
                return
            # Check if scheme exists, if not create geometry
            if not hasattr(self.parent, "scheme") or self.parent.scheme is None:
                # Get model bounds for geometry dialog
                if not hasattr(self.parent, "point_manager") or len(self.parent.point_manager.points) == 0:
                    QMessageBox.warning(self.parent, "Forward Model", "No model geometry defined")
                    return
                points = self.parent.point_manager.points
                x_coords = [p["x"] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                # Show geometry creation dialog
                geom_dialog = GeometryDialog(x_min, x_max, self.parent)
                if geom_dialog.exec_() != geom_dialog.Accepted:
                    return
                geom_params = geom_dialog.get_parameters()
                # Create scheme from geometry parameters
                self._create_scheme_from_geometry(geom_params)
                if self.parent.scheme is None:
                    QMessageBox.warning(self.parent, "Forward Model", "Failed to create geometry")
                    return
            # Run forward model
            response = self.parent.forward_model.run_model(
                self.parent.body_manager.bodies,
                self.parent.point_manager.points,
                self.parent.line_manager.lines,
                self.parent.scheme
            )
            self.response = response
            # If geometry was created and user requested save, write calculated times to picks.sgt
            if hasattr(self.parent, "save_pick_file") and self.parent.save_pick_file:
                try:
                    # Load tmp.sgt, update times, save as picks.sgt
                    if os.path.exists("tmp.sgt"):
                        updated_scheme = tt.load("tmp.sgt", verbose=False)
                        updated_scheme.set('t', response)
                        updated_scheme.save("picks.sgt")
                        # Update parent scheme to use the one with calculated times
                        if hasattr(self.parent, "set_scheme"):
                            self.parent.set_scheme(updated_scheme)
                        else:
                            self.parent.scheme = updated_scheme
                            if hasattr(self.parent, "get_pick_positions"):
                                self.parent.get_pick_positions()
                        # Plot calculated times as pseudo-measured picks
                        if hasattr(self.parent, "ax_dat") and hasattr(self.parent, "plot_picks"):
                            try:
                                self.parent.ax_dat.cla()
                                self.parent.ax_dat.set_title("Data plot")
                                self.parent.ax_dat.set_ylabel("Time [s]")
                                self.parent.plot_picks(self.parent.ax_dat, updated_scheme["t"], marker='+')
                            except Exception as e:
                                print(f"Warning: Could not plot picks: {e}")
                        if hasattr(self.parent, "statusBar"):
                            self.parent.statusBar().showMessage("Calculated times saved to picks.sgt", 3000)
                    # Reset flag
                        self.parent.save_geometry_to_picks = False
                    elif os.path.exists("picks.sgt"):
                        updated_scheme = tt.load("picks.sgt", verbose=False)
                        updated_scheme.set('t', response)
                        updated_scheme.save("picks.sgt")
                        self.parent.set_scheme(updated_scheme)
                        if hasattr(self.parent, "ax_dat") and hasattr(self.parent, "plot_picks"):
                            try:
                                self.parent.ax_dat.cla()
                                self.parent.ax_dat.set_title("Data plot")
                                self.parent.ax_dat.set_ylabel("Time [s]")
                                self.parent.plot_picks(self.parent.ax_dat, updated_scheme["t"], marker='+')
                            except Exception as e:
                                print(f"Warning: Could not plot picks: {e}")
                        if hasattr(self.parent, "statusBar"):
                            self.parent.statusBar().showMessage("Calculated times saved to picks.sgt", 3000)
                    
                except Exception as e:
                    print(f"Warning: Could not save to picks.sgt: {e}")
            # if os.path.exists("picks.sgt"):
            #     new_scheme = tt.load("picks.sgt", verbose=False)
            # elif os.path.exists("tmp.sgt"):
            #     new_scheme = tt.load("tmp.sgt", verbose=False)
            # else:
            #     new_scheme = updated_scheme
            times = np.array(self.parent.scheme["t"])
            new_scheme = self.parent.scheme
            new_scheme.set('t', response)
            new_scheme.save("forward_picks.sgt")
            self.parent.scheme.set("t", times)
            self.scheme_f = self.parent.scheme.copy()
            # Plot calculated travel times
            if hasattr(self.parent, "plot_calculated_times"):
                self.parent.plot_calculated_times(response)
                # Calculate and display chi2 and RMS
                if hasattr(self.parent, "scheme") and self.parent.scheme is not None:
                    meas_tt = np.array(self.parent.scheme["t"])
                    calc_tt = np.array(response)
                    # Calculate chi2 using data errors if available
                    if "err" in self.parent.scheme.dataMap():
                        errors = np.array(self.parent.scheme["err"])
                        chi2 = np.sum(((meas_tt - calc_tt) / errors)**2) / len(meas_tt)
                    else:
                        # If no errors, use simple normalized misfit
                        chi2 = np.sum(((meas_tt - calc_tt) / meas_tt)**2) / len(meas_tt)
                    rms = np.sqrt(np.mean((meas_tt - calc_tt)**2))
                    self.chi2_forward= chi2
                    self.rms_forward = rms
                    # Update title with chi2 and RMS
                    if hasattr(self.parent, "update_data_plot_title"):
                        self.parent.update_data_plot_title(chi2, rms)
                        self.parent.fig.savefig("forward_model_plot.png", dpi=300)
            # Enable View Mesh menu item
            self.view_mesh_action.setEnabled(True)
            self.toggle_rays_action.setEnabled(True)
            if not hasattr(self.parent, "save_geometry_to_picks") or not self.parent.save_geometry_to_picks:
                if hasattr(self.parent, "statusBar"):
                    self.parent.statusBar().showMessage("Forward model calculated successfully", 3000)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self.parent, "Forward Model", f"Failed to run forward model: {e}")
    
    def _create_scheme_from_geometry(self, params):
        """Create a DataContainer scheme from geometry parameters."""
        import numpy as np
        import pygimli as pg
        import pygimli.physics.traveltime as tt
        
        # Create receiver positions
        rx_x = np.arange(params['rx_count']) * params['rx_spacing'] + params['rx_start']
        
        # Create shot positions
        shot_x = np.arange(params['shot_count']) * params['shot_spacing'] + params['shot_start']
        
        # Combine all unique sensor positions (receivers + shots)
        all_x = np.unique(np.concatenate([rx_x, shot_x]))
        all_pos = np.column_stack([all_x, np.zeros_like(all_x)])
        
        # Create DataContainer
        scheme = pg.DataContainer()
        scheme.registerSensorIndex('s')  # shot index
        scheme.registerSensorIndex('g')  # geophone/receiver index
        
        # Add all unique sensor positions
        for pos in all_pos:
            scheme.createSensor(pos)
        
        # Create mapping from coordinates to sensor indices
        coord_to_idx = {x: i for i, x in enumerate(all_x)}
        
        # Build shot and receiver index arrays
        shot_indices = []
        receiver_indices = []
        for shot in shot_x:
            for rx in rx_x:
                shot_indices.append(coord_to_idx[shot])
                receiver_indices.append(coord_to_idx[rx])
        
        # Set up data entries
        n_data = len(shot_indices)
        scheme.resize(n_data)
        scheme.set('s', np.array(shot_indices, dtype=int))
        scheme.set('g', np.array(receiver_indices, dtype=int))
        scheme.set('t', np.zeros(n_data))  # Dummy travel times
        scheme.set('err', np.ones(n_data) * 0.0005)  # Dummy errors
        
        # Always save to tmp.sgt for use in forward model
        try:
            scheme.save("tmp.sgt")
        except Exception as e:
            QMessageBox.warning(self.parent, "Save Geometry", f"Failed to save temporary geometry: {e}")
            return
        
        # Store flag for later saving to picks.sgt after forward calculation
        self.parent.save_geometry_to_picks = params.get('save_to_file', True)
        
        # Load scheme from tmp.sgt to ensure consistency
        try:
            scheme = tt.load("tmp.sgt", verbose=False)
        except Exception as e:
            QMessageBox.warning(self.parent, "Load Geometry", f"Failed to load temporary geometry: {e}")
            return
        
        # Store the scheme and initialize pick positions
        if hasattr(self.parent, "set_scheme"):
            self.parent.set_scheme(scheme)
        else:
            self.parent.scheme = scheme
            if hasattr(self.parent, "get_pick_positions"):
                self.parent.get_pick_positions()
        
        # Plot the geometry on the data axis with dummy times
        if hasattr(self.parent, "plot_picks") and hasattr(self.parent, "ax_dat"):
            try:
                self.parent.ax_dat.cla()
                self.parent.ax_dat.set_title("Data plot")
                self.parent.ax_dat.set_ylabel("Time [s]")
                # Plot with dummy zero times
                self.parent.plot_picks(self.parent.ax_dat, scheme["t"], marker='+')
            except Exception as e:
                print(f"Warning: Could not plot picks: {e}")

    def select_picks(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._select_picks_impl)
        return self._select_picks_impl()

    def _select_picks_impl(self):
        if hasattr(self.parent, "select_picks"):
            self.parent.select_picks()

    def set_inversion_parameters(self):
        if hasattr(self.parent, 'handle_menu_action'):
            return self.parent.handle_menu_action(self._set_inversion_parameters_impl)
        return self._set_inversion_parameters_impl()

    def _set_inversion_parameters_impl(self):
        """Open dialog to set inversion parameters."""
        dialog = InversionParametersDialog(self.parent)
        if dialog.exec_():
            params = dialog.get_parameters()
            # Store parameters in parent for use during inversion
            self.parent.inversion_params = params
            # Show confirmation
            start_model = "Actual model" if params['use_actual_model'] else "Gradient model"
            if hasattr(self.parent, "statusBar"):
                self.parent.statusBar().showMessage(
                    f"Inversion parameters set: Starting model = {start_model}", 
                    3000
                )
            # Handle body regularization if requested
            if params.get('modify_regularization', False):
                self._handle_body_regularization()
            elif params.get('do_inversion', False):
                # Run inversion directly if no body modifications needed
                self.run_inversion()
    
    def _handle_body_regularization(self):
        """Allow user to click on bodies and set regularization parameters."""
        from PyQt5.QtWidgets import QMessageBox
        
        print("*** Starting body regularization mode ***")
        
        # Initialize storage for body regularization settings
        if not hasattr(self.parent, 'body_regularization'):
            self.parent.body_regularization = {}
        
        # Connect canvas click event and key press event
        self.parent._body_reg_mode = True
        print("Connecting button_press_event handler...")
        self.parent._body_reg_cid = self.parent.fig.canvas.mpl_connect(
            'button_press_event', self._on_body_click_for_regularization
        )
        print(f"Connected with cid={self.parent._body_reg_cid}")
        
        self.parent._body_reg_key_cid = self.parent.fig.canvas.mpl_connect(
            'key_press_event', self._on_key_press_for_regularization
        )
        print(f"Connected key handler with cid={self.parent._body_reg_key_cid}")
        
        # Show instructions
        QMessageBox.information(
            self.parent, 
            "Body Regularization",
            "Click on bodies to set regularization parameters.\n\n"
            "Press ENTER when finished or ESC to cancel."
        )
        
        if hasattr(self.parent, "statusBar"):
            self.parent.statusBar().showMessage(
                "Click on bodies to modify regularization. Press ENTER when done or ESC to cancel.", 
                0
            )
        
        # Show overlay message
        if hasattr(self.parent, 'show_overlay_message'):
            self.parent.show_overlay_message("BODY REGULARIZATION MODE\nClick bodies to edit | Press ENTER when done | Press ESC to cancel")
        
        print("Body regularization mode ready - waiting for clicks...")
    
    def _on_body_click_for_regularization(self, event):
        """Handle click on canvas to select a body for regularization."""
        from matplotlib.path import Path
        
        print(f"Click event: inaxes={event.inaxes}, button={event.button}, x={event.xdata}, y={event.ydata}")
        
        if event.inaxes != self.parent.ax:
            print("Click not in ax, ignoring")
            return
        
        if event.dblclick:  # Only respond to single clicks
            print("Double-click detected, ignoring")
            return
            
        x, y = event.xdata, event.ydata
        print(f"Processing click at ({x:.2f}, {y:.2f})")
        
        # Find which body was clicked
        bodies = self.parent.body_manager.bodies
        points = self.parent.point_manager.points
        lines = self.parent.line_manager.lines
        
        for body_idx, body in enumerate(bodies):
            # Build polygon for this body
            poly_points = []
            for j, line_idx in enumerate(body["lines"]):
                line = lines[line_idx]
                if j == 0:
                    if body["sense"][j] > 0:
                        p = points[line["point1"]]
                        poly_points.append([p["x"], p["y"]])
                    else:
                        p = points[line["point2"]]
                        poly_points.append([p["x"], p["y"]])
                
                if body["sense"][j] > 0:
                    p = points[line["point2"]]
                else:
                    p = points[line["point1"]]
                poly_points.append([p["x"], p["y"]])
            
            if len(poly_points) >= 3:
                poly_points = poly_points[:-1]  # Remove duplicate
                path = Path(poly_points)
                
                if path.contains_point([x, y]):
                    # Found the body - show dialog
                    print(f"Body {body_idx + 1} clicked")
                    velocity = body["props"][0]
                    dialog = BodyRegularizationDialog(body_idx, velocity, self.parent)
                    
                    if dialog.exec_():
                        params = dialog.get_parameters()
                        self.parent.body_regularization[body_idx] = params
                        
                        if hasattr(self.parent, "statusBar"):
                            self.parent.statusBar().showMessage(
                                f"Body {body_idx + 1} regularization updated: {params['type']}", 
                                3000
                            )
                    break
    
    def _on_key_press_for_regularization(self, event):
        """Handle key press during body regularization mode."""
        if event.key == 'enter' or event.key == 'return':
            # Hide overlay message immediately
            if hasattr(self.parent, 'hide_overlay_message'):
                self.parent.hide_overlay_message()
            # Force canvas redraw and process events
            self.parent.fig.canvas.draw()
            QApplication.processEvents()
            # Use timer to delay inversion start slightly to ensure message is gone
            QTimer.singleShot(50, lambda: self._finish_body_regularization_mode(proceed=True))
        elif event.key == 'escape':
            # Cancel
            self._finish_body_regularization_mode(proceed=False)
    
    def _finish_body_regularization_mode(self, proceed=True):
        """Finish body regularization mode."""
        # Hide overlay message
        if hasattr(self.parent, 'hide_overlay_message'):
            self.parent.hide_overlay_message()
        
        # Disconnect event handlers
        if hasattr(self.parent, '_body_reg_cid'):
            self.parent.fig.canvas.mpl_disconnect(self.parent._body_reg_cid)
            delattr(self.parent, '_body_reg_cid')
        
        if hasattr(self.parent, '_body_reg_key_cid'):
            self.parent.fig.canvas.mpl_disconnect(self.parent._body_reg_key_cid)
            delattr(self.parent, '_body_reg_key_cid')
        
        self.parent._body_reg_mode = False
        
        if hasattr(self.parent, "statusBar"):
            self.parent.statusBar().clearMessage()
        
        if proceed:
            # Run inversion with body regularization settings
            self.run_inversion()
        else:
            # Clear body regularization settings
            if hasattr(self.parent, 'body_regularization'):
                self.parent.body_regularization = {}
    
    def run_inversion(self):
        from PyQt5.QtWidgets import QMessageBox
        try:
            import os
            import pygimli.physics.traveltime as tt
            
            # Check if inversion parameters are set
            if not hasattr(self.parent, "inversion_params"):
                QMessageBox.warning(self.parent, "Inversion", 
                                       "Please set inversion parameters first (F8)")
                return
            
            # Check if picks file exists and load it
            if not hasattr(self.parent, "scheme") or self.parent.scheme is None:
                # Try to load picks from file
                if os.path.exists("picks.sgt"):
                    try:
                        self.parent.scheme = tt.load("picks.sgt", verbose=True)
                        if hasattr(self.parent, "statusBar"):
                            self.parent.statusBar().showMessage("Loaded picks.sgt for inversion", 2000)
                    except Exception as e:
                        QMessageBox.warning(self.parent, "Inversion", f"Failed to load picks.sgt: {e}")
                        return
                elif os.path.exists("picks.dat"):
                    try:
                        self.parent.scheme = tt.load("picks.dat", verbose=True)
                        if hasattr(self.parent, "statusBar"):
                            self.parent.statusBar().showMessage("Loaded picks.dat for inversion", 2000)
                    except Exception as e:
                        QMessageBox.warning(self.parent, "Inversion", f"Failed to load picks.dat: {e}")
                        return
                else:
                    # No picks.sgt or picks.dat found - ask user to select a .sgt file or create geometry
                    from PyQt5.QtWidgets import QFileDialog, QMessageBox
                    
                    reply = QMessageBox.question(
                        self.parent,
                        "No picks file found",
                        "picks.sgt not found.\n\nDo you want to:\n- Select an existing .sgt file (Yes)\n- Create geometry for virtual receivers/shots (No)",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        # User wants to select an existing file
                        file_path, _ = QFileDialog.getOpenFileName(
                            self.parent,
                            "Select picks file",
                            "",
                            "Seismic data files (*.sgt);;All files (*.*)"
                        )
                        
                        if file_path:
                            try:
                                self.parent.scheme = tt.load(file_path, verbose=True)
                                if hasattr(self.parent, "statusBar"):
                                    self.parent.statusBar().showMessage(f"Loaded {os.path.basename(file_path)} for inversion", 2000)
                            except Exception as e:
                                QMessageBox.warning(self.parent, "Inversion", f"Failed to load {file_path}: {e}")
                                return
                        else:
                            # User cancelled file selection
                            return
                    
                    elif reply == QMessageBox.No:
                        # User wants to create geometry - get model bounds
                        if not hasattr(self.parent, "point_manager") or len(self.parent.point_manager.points) == 0:
                            QMessageBox.warning(self.parent, "Inversion", "No model geometry defined")
                            return
                        
                        points = self.parent.point_manager.points
                        x_coords = [p["x"] for p in points]
                        x_min, x_max = min(x_coords), max(x_coords)
                        
                        # Show geometry creation dialog
                        from ui.dialogs import GeometryDialog
                        geom_dialog = GeometryDialog(x_min, x_max, self.parent)
                        if geom_dialog.exec_() != geom_dialog.Accepted:
                            return
                        
                        geom_params = geom_dialog.get_parameters()
                        
                        # Create scheme from geometry parameters
                        self._create_scheme_from_geometry(geom_params)
                        
                        if self.parent.scheme is None:
                            QMessageBox.warning(self.parent, "Inversion", "Failed to create geometry")
                            return
                    
                    else:
                        # User cancelled
                        return
            
            # Check if we have bodies
            if not hasattr(self.parent, "body_manager") or len(self.parent.body_manager.bodies) == 0:
                QMessageBox.warning(self.parent, "Inversion", "No bodies defined in the model")
                return
            
            # Check if forward model exists (for actual model case)
            if self.parent.inversion_params['use_actual_model']:
                if not hasattr(self.parent, "forward_model") or self.parent.forward_model.mesh is None:
                    QMessageBox.warning(self.parent, "Inversion", 
                                       "Please run forward model first (F5) when using actual model as starting model")
                    return
            
            # Run inversion
            if hasattr(self.parent, "statusBar"):
                self.parent.statusBar().showMessage("Running inversion...", 0)
            
            if hasattr(self.parent, "inversion"):
                # Get body regularization settings if they exist
                body_reg = getattr(self.parent, 'body_regularization', {})
                
                # Debug: Show if any body regularization is active
                if body_reg:
                    print("\n*** Active body regularization settings: ***")
                    for body_idx, params in body_reg.items():
                        print(f"  Body {body_idx}: {params}")
                    print("*** To clear these settings, run 'Modify regularization in specific bodies' and press ESC ***\n")
                
                vest, mgr, scheme = self.parent.inversion.run_inversion(
                    self.parent.inversion_params,
                    self.parent.scheme,
                    self.parent.body_manager.bodies,
                    self.parent.point_manager.points,
                    self.parent.line_manager.lines,
                    self.parent.forward_model,
                    body_reg
                )
                self.vest = vest
                self.mgr = mgr
                self.scheme_inv = scheme.copy()
                self.parent.polygone_fill_flag = False
                
                # Update scheme with filtered version (zero times removed)
                self.parent.set_scheme(scheme)
                
                # Plot calculated travel times in upper canvas
                if hasattr(self.parent, "plot_calculated_times") and hasattr(self.parent, "ax_dat"):
                    self.calc_times = np.array(mgr.inv.response)
                    self.parent.plot_calculated_times(self.calc_times)
                    
                    # Update title with chi2 and RMS
                    chi2 = mgr.inv.chi2()
                    meas_tt = np.array(scheme["t"])
                    rms = np.sqrt(np.mean((meas_tt - self.calc_times)**2))
                    self.parent.update_data_plot_title(chi2, rms)
                
                # Update title with chi2 and RMS
                chi2 = mgr.inv.chi2()
                meas_tt = np.array(scheme["t"])
                rms = np.sqrt(np.mean((meas_tt - self.calc_times)**2))
                self.rms_inv = rms
                self.chi2_inv = chi2
                self.parent.update_data_plot_title(chi2, rms)
                
                # Plot inverted model in lower canvas
                print("\n*** Initial inversion complete - plotting to canvas ***", flush=True)
                print(f"parent has ax: {hasattr(self.parent, 'ax')}", flush=True)
                if hasattr(self.parent, "ax"):
                    print("Calling _plot_inverted_model_to_canvas for initial result...", flush=True)
                    self._plot_inverted_model_to_canvas(vest, mgr, self.parent)
                    self.parent.show_inverted = True  # Set flag so toggle works correctly
                    print("Initial canvas plot completed", flush=True)
                else:
                    print("ERROR: No ax attribute on parent!", flush=True)
                
                # Show chi2 evolution and ask for additional iterations
                self._show_chi2_and_continue(mgr, vest, scheme)
            
            # Enable Toggle Model Plot, Toggle Rays, and Color Scale menu items after inversion
            self.toggle_plot_action.setEnabled(True)
            self.toggle_rays_action.setEnabled(True)
            self.color_scale_action.setEnabled(True)
            
            if hasattr(self.parent, "statusBar"):
                self.parent.statusBar().showMessage("Inversion completed successfully", 3000)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self.parent, "Inversion", f"Failed to run inversion: {e}")

    # --- View actions ---
    def toggle_plot(self):
        if hasattr(self.parent, "toggle_model_plot_mode"):
            self.parent.toggle_model_plot_mode()
    
    def toggle_rays(self):
        """Toggle ray path visibility on the model plot."""
        if hasattr(self.parent, 'toggle_ray_plot'):
            self.parent.toggle_ray_plot()
    
    def view_mesh(self):
        """View the mesh with regions from the forward model."""
        if hasattr(self.parent, "forward_model") and self.parent.forward_model.mesh is not None:
            self.parent.forward_model.plot_mesh()
        else:
            QMessageBox.information(self.parent, "View Mesh", 
                                   "No mesh available. Run forward model first (F5).")
    
    def configure_color_scale(self):
        """Configure color scale for inverted model display."""
        if not hasattr(self.parent, 'inversion') or self.parent.inversion.vest is None:
            QMessageBox.information(self.parent, "Color Scale", 
                                   "No inversion results available. Run inversion first (F9).")
            return
        
        # Get current settings
        current_vmin = getattr(self.parent, '_color_vmin', None)
        current_vmax = getattr(self.parent, '_color_vmax', None)
        current_cmap = getattr(self.parent, '_color_cmap', 'self.parent.cmap')
         
        # Show dialog
        dialog = ColorScaleDialog(current_vmin, current_vmax, current_cmap, self.parent)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            
            # Store settings
            self.parent._color_vmin = params['vmin']
            self.parent._color_vmax = params['vmax']
            self.parent._color_cmap = params['cmap']
            self.parent.cmap =  mpl.colormaps[params['cmap']]
            print(f"Dialog: {hasattr(self.parent, 'plot_inverted_model')}")
            
            # Replot the inverted model with new color scale
            # if hasattr(self.parent, 'plot_inverted_model'):
            if hasattr(self.parent, 'plot_inverted_model') and self.parent.show_inverted:
                print("Plot inverted model with new color scale")
                self.parent.plot_inverted_model()
            elif hasattr(self.parent,'forward_model') and not self.parent.show_inverted:
                print("Plot forward model with new color scale")
                self.parent.show_inverted= True
                self.parent.toggle_model_plot_mode()
            
            if hasattr(self.parent, "statusBar"):
                self.parent.statusBar().showMessage("Color scale updated", 2000)
    
    def clear_body_regularization(self):
        """Clear all body-specific regularization settings."""
        if hasattr(self.parent, 'body_regularization'):
            if self.parent.body_regularization:
                count = len(self.parent.body_regularization)
                self.parent.body_regularization = {}
                QMessageBox.information(self.parent, "Body Regularization", 
                                       f"Cleared regularization settings for {count} bod{'y' if count == 1 else 'ies'}.\n"
                                       f"Next inversion will use default settings.")
                if hasattr(self.parent, "statusBar"):
                    self.parent.statusBar().showMessage(f"Cleared body regularization for {count} bod{'y' if count == 1 else 'ies'}", 3000)
            else:
                QMessageBox.information(self.parent, "Body Regularization", 
                                       "No body regularization settings are currently active.")
        else:
            self.parent.body_regularization = {}
            QMessageBox.information(self.parent, "Body Regularization", 
                                   "No body regularization settings are currently active.")
    
    def about_dialog(self):
        QMessageBox.information(
            self.parent,
            "About Geological Model Builder",
            f"Geological Model Builder version {version.__version__}\n\n"
            "© 2025 Hermann Zeyen, University Paris-Saclay"
        )

    def help_dialog(self):
        QMessageBox.information(
            self.parent,
            "Option of Geological Model Builder",
            "Actions on bodies: 'Edit -> Body Editor or press B)\n"
            "   Split body(ies) in two parts: left mouse and pull line\n"
            "   Join two bodies: Right click into first body followed by "
            "click into second body\n"
            "                    Joint body has properties of first clicked body\n\n"
            "Actions on nodes: 'Edit -> Node Editor or press N)\n"
            "   Move node:    Left click near node and pull to new position\n"
            "   Delete node:  Left click near node followed by keyboard DEL\n"
            "   Add new node: Right click near edge where to add the node\n\n"
            "Change properties: 'Edit -> Property Editor or press P)\n"
            "   Click into body to call dialog box to define properties\n\n"
            "Define picks to be used: 'Tools -> Select picks or press F4)\n\n"
            "   ALL THESE EDITING OPTIONS MUST BE FINISHED PRESSING\n"
                    "ENTER (accept) or ESC (reject all modifs since last call)"
            "Run forward calculation: 'Tools -> Run Forward Model or press F4)\n\n"
            "Run inversion:\n"
            "   IMPORTANT: Before running an inversion, some parameters must be set:\n"
            "              'Tools -> Inversion Parameters or press F8\n"
            "   'Tools -> Run Inversion or press F9 or check box in dialog "
            "box Inversion Parameters\n\n"
            "Set Inversion parameters 'Tools -> Inversion Parameters or press F8\n"
            "   You may start the inversion with a simple gradient model or\n"
            "   using the actual interactive model.\n"
            "   When using the gradient model, fill in the fields,\n"
            "        check whether the inversion should be run immediately and accept\n"
            "   When using the actual model, you may set specific regulatization\n"
            "        for every single body\n"
            "   SEE MANUAL FOR MORE INFORMATION\n\n"
        )
        
    
    def _plot_inverted_model_to_canvas(self, vest, mgr, parent):
        """Plot inverted model to the lower canvas with body contours."""
        import pygimli as pg
        
        print("\n=== _plot_inverted_model_to_canvas called ===", flush=True)
        print(f"vest type: {type(vest)}, len/shape: {vest.shape if hasattr(vest, 'shape') else len(vest)}", flush=True)
        print(f"mgr type: {type(mgr)}", flush=True)
        print(f"parent has ax: {hasattr(parent, 'ax')}", flush=True)
        print(f"parent has canvas: {hasattr(parent, 'canvas')}", flush=True)
        
        # Clear the model axis
        print("Clearing ax...", flush=True)
        parent.ax.cla()
        
        # Get velocity bounds (use custom settings if available)
        print("Calculating velocity bounds...", flush=True)
        if hasattr(parent, '_color_vmin') and parent._color_vmin is not None:
            vmin = parent._color_vmin
        else:
            vmin = np.quantile(vest, 0.01)
        
        if hasattr(parent, '_color_vmax') and parent._color_vmax is not None:
            vmax = parent._color_vmax
        else:
            vmax = np.quantile(vest, 0.99)
        
        # Get colormap (use custom setting if available)
        cmap = getattr(parent, '_color_cmap', 'jet')
        
        print(f"vmin={vmin:.1f}, vmax={vmax:.1f}, cmap={cmap}", flush=True)
        
        # Plot inverted model
        print("Calling pg.show...", flush=True)
        try:
            # Clear colorbar axis
            if hasattr(parent, 'ax_cb'):
                parent.ax_cb.cla()
            
            # Use the forward operator mesh (the actual inversion mesh with all cells)
            mesh_to_plot = mgr.fop.mesh()
            pg.show(mesh_to_plot, data=vest, ax=parent.ax, 
                    cMin=vmin, cMax=vmax, cMap=cmap, 
                    logScale=False, colorBar=False)  # Don't create inline colorbar
            
            # Add colorbar to dedicated axis
            if hasattr(parent, 'ax_cb'):
                from matplotlib import cm
                import matplotlib.pyplot as plt
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = parent.fig.colorbar(sm, cax=parent.ax_cb)
                cbar.set_label('Velocity [m/s]')
            
            print("pg.show completed", flush=True)
        except Exception as e:
            print(f"ERROR in pg.show: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return
        
        # Plot body contours on top
        if hasattr(parent, "body_manager"):
            for body in parent.body_manager.bodies:
                x_coords = []
                y_coords = []
                for j, line_idx in enumerate(body["lines"]):
                    line = parent.line_manager.lines[line_idx]
                    if j == 0:
                        if body["sense"][j] > 0:
                            p = parent.point_manager.points[line["point1"]]
                            x_coords.append(p["x"])
                            y_coords.append(p["y"])
                    
                    if body["sense"][j] > 0:
                        p = parent.point_manager.points[line["point2"]]
                    else:
                        p = parent.point_manager.points[line["point1"]]
                    x_coords.append(p["x"])
                    y_coords.append(p["y"])
                
                # Plot body boundary
                parent.ax.plot(x_coords, y_coords, 'k-', linewidth=1.5)
        
        parent.ax.set_xlabel("Distance [m]")
        parent.ax.set_ylabel("Depth [m]")
        parent.ax.set_title("Inverted Velocity Model")
        
        # Set xlim to match the upper data plot
        if hasattr(parent, 'ax_dat'):
            xlim = parent.ax_dat.get_xlim()
            parent.ax.set_xlim(xlim)
            print(f"Set xlim: [{xlim[0]:.1f}, {xlim[1]:.1f}]", flush=True)
        
        # Force immediate canvas update
        try:
            parent.canvas.draw()
            parent.canvas.flush_events()
            parent.fig.canvas.draw_idle()
            QApplication.processEvents()
            print("Inverted model plotted to lower canvas", flush=True)
        except Exception as e:
            print(f"Error drawing canvas: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _show_chi2_and_continue(self, mgr, vest, scheme, cumulative_chi2=None):
        """Show chi2 evolution and ask for additional iterations."""
        # Get chi2 history (cumulative across all iterations)
        if cumulative_chi2 is None:
            chi_history = np.array(mgr.inv.chi2History)
        else:
            # Append new iterations to cumulative history
            chi_history = cumulative_chi2
        
        # Create chi2 plot window
        fig, ax = plt.subplots(figsize=(8, 6))
        iterations = np.arange(len(chi_history)) + 1
        ax.plot(iterations, np.log10(chi_history), 'b-o', markersize=6)
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("log₁₀(Chi²)")
        ax.set_title(f"Chi² Evolution (Final Chi²: {chi_history[-1]:.3f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Position figure window on same screen as main window
        figManager = plt.get_current_fig_manager()
        if hasattr(self.parent, 'geometry'):
            main_geom = self.parent.geometry()
            # Position chi2 plot near main window
            figManager.window.setGeometry(main_geom.x() + 100, main_geom.y() + 100, 800, 600)
        
        # Make plot window stay on top
        figManager.window.setWindowFlags(figManager.window.windowFlags() | Qt.WindowStaysOnTopHint)
        plt.show(block=False)
        figManager.window.activateWindow()
        figManager.window.raise_()
        QApplication.processEvents()  # Process events to show window
        plt.pause(0.2)  # Give it time to appear
        
        # Show dialog with chi2 info
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Inversion Complete")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Inversion Results:</b>"))
        layout.addWidget(QLabel(f"Final Chi²: {chi_history[-1]:.3f}"))
        layout.addWidget(QLabel(f"Total iterations: {len(chi_history)}"))
        layout.addWidget(QLabel(""))
        layout.addWidget(QLabel("To run more iterations:"))
        layout.addWidget(QLabel("1. Close this dialog"))
        layout.addWidget(QLabel("2. Press F8 (Inversion Parameters)"))
        layout.addWidget(QLabel("3. Increase 'Maximum number of iterations'"))
        layout.addWidget(QLabel("4. Check 'Do inversion' and click OK"))
        
        button_layout = QHBoxLayout()
        finish_button = QPushButton("Close")
        button_layout.addWidget(finish_button)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        finish_button.clicked.connect(dialog.accept)
        
        _ = dialog.exec_()
        
        plt.close(fig)
        
        # Save final inverted model plot to folder
        if hasattr(self.parent, "inversion") and hasattr(self.parent.inversion, "_plot_inverted_model"):
            xmin = min(scheme.sensors().array()[:, 0])
            xmax = max(scheme.sensors().array()[:, 0])
            self.parent.inversion._plot_inverted_model(xmin, xmax)
        
        # Plot to canvas
        if hasattr(self.parent, "ax"):
            self._plot_inverted_model_to_canvas(vest, mgr, self.parent)
            self.parent.show_inverted = True  # Set flag so toggle works correctly
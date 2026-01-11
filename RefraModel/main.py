"""
Main entry point for RefraModel application
"""
import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import numpy as np

from .model_builder import ModelBuilder
from .ui.dialogs import StartParamsDialog


def _create_geometry_and_times(window, geom_params):
    """Create acquisition geometry and calculate travel times using forward model"""
    import pygimli.physics.traveltime as tt
    import numpy as np
    
    # Create receivers
    rx_start = geom_params['rx_start']
    rx_spacing = geom_params['rx_spacing']
    rx_count = geom_params['rx_count']
    
    # Create shots
    shot_start = geom_params['shot_start']
    shot_spacing = geom_params['shot_spacing']
    shot_count = geom_params['shot_count']
    
    # Build receiver positions
    rx_x = np.array([rx_start + i * rx_spacing for i in range(rx_count)])
    
    # Build shot positions
    shot_x = np.array([shot_start + i * shot_spacing for i in range(shot_count)])
    
    # Combine all unique sensor positions (receivers + shots)
    all_x = np.unique(np.concatenate([rx_x, shot_x]))
    all_pos = np.column_stack([all_x, np.zeros_like(all_x)])
    
    # Create DataContainer for traveltime data
    scheme = tt.DataContainer()
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
    
    # Set scheme in window
    window.set_scheme(scheme)
    
    # Calculate travel times using forward model if bodies exist
    if hasattr(window, "body_manager") and len(window.body_manager.bodies) > 0:
        try:
            scheme.set('t', np.zeros(n_data))  # Temporary zeros for forward model
            response = window.forward_model.run_model(
                window.body_manager.bodies,
                window.point_manager.points,
                window.line_manager.lines,
                scheme
            )
            
            # Update scheme with calculated times and errors
            scheme.set('t', response)
            scheme.set('err', np.ones(n_data) * 0.0001)  # Dummy errors (0.1 ms)
            
            # Save to picks.sgt if requested
            if geom_params.get('save_to_file', False):
                scheme.save("picks.sgt")
                print("Saved geometry and calculated times to picks.sgt")
        except Exception as e:
            print(f"Warning: Could not calculate travel times: {e}")
            # Create dummy times (zero) and errors
            scheme.set('t', np.zeros(n_data))
            scheme.set('err', np.ones(scheme.size()) * 0.0005)  # Dummy errors


def main():
    """Main entry point"""
    # Use current working directory as starting point (where user ran the command)
    working_dir = os.getcwd()
    print(f"Working directory: {working_dir}")
    
    app = QApplication(sys.argv)
    
    # Get all screens
    screens = app.screens()
    print(f"Detected {len(screens)} screen(s).")
    
    # Defaults for area and threshold
    xmin = 0.0
    xmax = 100.0
    ymin = -30.0
    ymax = 0.0
    threshold = 1.0

    # First, attempt to read picks.sgt if it exists (used for plotting always, and as defaults when no model file)
    scheme = None
    picks_defaults = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    picks_file = None
    
    if os.path.exists("picks.sgt"):
        picks_file = "picks.sgt"
    else:
        # picks.sgt not found - ask user if they want to select one
        reply = QMessageBox.question(
            None,
            "picks.sgt not found",
            "File picks.sgt not found in current directory.\n\n"
            "Do you want to select a picks file (*.sgt)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Start file dialog in current working directory
            picks_file, _ = QFileDialog.getOpenFileName(
                None,
                "Select picks file",
                working_dir,
                "Seismic data files (*.sgt);;All Files (*.*)"
            )
            # Note: Don't change working directory - picks file can be anywhere
    
    # Load picks file if available
    xtopo_for_model = []
    ytopo_for_model = []
    if picks_file:
        try:
            import pygimli.physics.traveltime as tt
            print(f"\nLoading picks file: {picks_file}")
            scheme = tt.load(picks_file, verbose=True)
            pos = np.array(scheme.sensors())
            
            # Check for topography (z-coordinates in third column)
            z_coords = pos[:, 2] if pos.shape[1] >= 3 else np.zeros(len(pos))
            unique_z = np.unique(z_coords)
            
            # Determine x-extents
            x_coords = pos[:, 0]
            picks_defaults["xmin"] = float(np.ceil(x_coords.min()))
            picks_defaults["xmax"] = float(np.ceil(x_coords.max()))
            
            # Check if topography exists (different z values)
            if len(unique_z) > 1:
                # Topography exists - normalize to highest point at 0
                max_z = z_coords.max()
                normalized_z = z_coords - max_z
                
                # Get unique x positions with their normalized z values
                xtopo = []
                ytopo = []
                for i in range(len(pos)):
                    xtopo.append(pos[i, 0])
                    ytopo.append(normalized_z[i])
                
                # Keep only unique x positions
                xx, index = np.unique(np.array(xtopo), return_index=True)
                if len(xx) > 1:
                    xtopo_for_model = np.copy(xx)
                    ytopo_for_model = np.copy(np.array(ytopo)[index])
                    
                    # Y_max should be 0 (highest point)
                    picks_defaults["ymax"] = 0.0
                    
                    # Y_min based on depth from lowest topography point
                    topo_min = ytopo_for_model.min()  # Most negative (deepest) topography
                    depth_extent = (picks_defaults["xmax"] - picks_defaults["xmin"]) * 0.3
                    picks_defaults["ymin"] = float(round(topo_min - depth_extent))
            else:
                # No topography - all z values the same (flat surface at 0)
                picks_defaults["ymax"] = 0.0
                picks_defaults["ymin"] = float(-round((picks_defaults["xmax"] - picks_defaults["xmin"]) * 0.3))
                
        except Exception as e:
            print(f"Could not read picks file: {e}")
            scheme = None

    # Prompt for existing model file (optional) then parameters
    # Start file dialog in current directory (dir0 if set, otherwise current)
    model_file, _ = QFileDialog.getOpenFileName(None, "Choose model file", os.getcwd(), "Model/Text Files (*.txt *.csv *.mod);;All Files (*.*)")

    # If a model file was chosen, change working directory to its location
    if model_file:
        model_dir = os.path.dirname(os.path.abspath(model_file))
        os.chdir(model_dir)
        print(f"Working directory changed to: {model_dir}")

    xmin = 0.0
    xmax = 100.0
    ymin = -30.0
    ymax = 0.0
    threshold = 1.0
    
    if model_file:
        # Ask only for threshold
        from os.path import basename
        dlg = StartParamsDialog(threshold_only=True, defaults={"threshold": threshold, "save_file": basename(model_file)})
        if dlg.exec_() != dlg.Accepted:
            return
        vals = dlg.values()
        if vals:
            threshold = vals["threshold"]
            save_file = vals.get("save_file", basename(model_file))
        else:
            from os.path import basename as _bn
            save_file = _bn(model_file)
        # Backup chosen model file
        try:
            if os.path.isfile(model_file):
                shutil.copyfile(model_file, model_file + ".bak")
        except Exception as e:
            print(f"Could not backup model file: {e}")
        # Initialize with degenerate extents to trigger auto-extents from file
        window = ModelBuilder(screens, xmin=0.0, xmax=0.0, ymin=0.0, ymax=0.0, threshold_pct=threshold)
        # Save path for edits
        window.model_save_path = save_file
        window.load_model_from_file(model_file)
        
        # If no picks file, create geometry now (required for modeling)
        if scheme is None:
            # Get model bounds for geometry dialog
            if hasattr(window, "point_manager") and len(window.point_manager.points) > 0:
                points = window.point_manager.points
                x_coords = [p["x"] for p in points]
                x_min_geom, x_max_geom = min(x_coords), max(x_coords)
                
                # Show geometry creation dialog
                from .ui.dialogs import GeometryDialog
                geom_dialog = GeometryDialog(x_min_geom, x_max_geom, window)
                if geom_dialog.exec_() == geom_dialog.Accepted:
                    geom_params = geom_dialog.get_parameters()
                    
                    # Create scheme and calculate times using forward model
                    _create_geometry_and_times(window, geom_params)
                    scheme = window.scheme
        
        # Always plot picks if available
        if scheme is not None:
            window.set_scheme(scheme)
            try:
                data = scheme["t"]
            except Exception:
                data = None
            if data is not None:
                window.plot_picks(window.ax_dat, data, marker="+")
                # Integrate topography as surface if applicable and re-plot model
                window.integrate_topography_into_surface()
                window.plot_model()
    else:
        dlg = StartParamsDialog(
            threshold_only=False,
            defaults={
                "xmin": picks_defaults["xmin"],
                "xmax": picks_defaults["xmax"],
                "ymin": picks_defaults["ymin"],
                "ymax": picks_defaults["ymax"],
                "threshold": threshold,
                "nprops": 1,
                "save_file": "model.txt",
            },
        )
        if dlg.exec_() != dlg.Accepted:
            return
        vals = dlg.values()
        if vals:
            xmin = vals["xmin"]
            xmax = vals["xmax"]
            ymin = vals["ymin"]
            ymax = vals["ymax"]
            threshold = vals["threshold"]
            prop_names = vals.get("prop_names", ["velocity"])
            prop_values = vals.get("prop_values", [1500.0])
            save_file = vals.get("save_file", "model.txt")
        else:
            save_file = "model.txt"
        window = ModelBuilder(screens, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, threshold_pct=threshold)
        window.model_save_path = save_file
        
        # Set topography data if available from picks file
        if len(xtopo_for_model) > 0 and len(ytopo_for_model) > 0:
            window.xtopo = xtopo_for_model
            window.ytopo = ytopo_for_model
            window.topo_flag = True
        
        # Always create the rectangular starting model in the no-model-file case
        window.start_model(prop_names=prop_names, prop_values=prop_values)
        
        # If no picks file, create geometry now (required for modeling)
        if scheme is None:
            # Show geometry creation dialog
            from .ui.dialogs import GeometryDialog
            geom_dialog = GeometryDialog(xmin, xmax, window)
            if geom_dialog.exec_() == geom_dialog.Accepted:
                geom_params = geom_dialog.get_parameters()
                
                # Create scheme and calculate times using forward model
                _create_geometry_and_times(window, geom_params)
                scheme = window.scheme
        
        # If picks were detected, attach scheme, plot picks, and integrate topography
        if scheme is not None:
            window.set_scheme(scheme)
            try:
                data = scheme["t"]
            except Exception:
                data = None
            if data is not None:
                window.plot_picks(window.ax_dat, data, marker="+")
            window.integrate_topography_into_surface()
            window.plot_model()
    window.show()
    
    if len(screens) > 1:
        second_screen = screens[1]
        geometry = second_screen.availableGeometry()
        print(f"Opening on second screen: {geometry}")
        window.move(geometry.topLeft())
        window.showMaximized()
    else:
        print("No secondary screen detected. Opening on primary screen.")
        window.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
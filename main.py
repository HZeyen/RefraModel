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


def main():
    """Main entry point"""
    dir0 = r"E:\Seg2Dat\Fontaines-Salees\2020\Profil1\Forward"
    if os.path.exists(dir0):
        os.chdir(dir0)
    
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
            "File picks.sgt not found in current directory.\n\nDo you want to select a picks file (*.sgt)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Start file dialog in dir0 if it exists, otherwise current directory
            start_dir = dir0 if os.path.exists(dir0) else os.getcwd()
            picks_file, _ = QFileDialog.getOpenFileName(
                None,
                "Select picks file",
                start_dir,
                "Seismic data files (*.sgt);;All Files (*.*)"
            )
            # Note: Don't change working directory - picks file can be anywhere
    
    # Load picks file if available
    if picks_file:
        try:
            import pygimli.physics.traveltime as tt
            print(f"\nLoading picks file: {picks_file}")
            scheme = tt.load(picks_file, verbose=True)
            pos = np.array(scheme.sensors())
            # Determine topography if Z varies
            t = np.unique(pos[:, 2]) if pos.shape[1] >= 3 else np.array([0.])
            xtopo = []
            ytopo = []
            if len(t) > 1:
                for i, s in enumerate(np.array(scheme.sensors())):
                    xtopo.append(pos[i, 0])
                    ytopo.append(pos[i, 2])
                xx, index = np.unique(np.array(xtopo), return_index=True)
                if len(xx) > 1:
                    xtopo = np.copy(xx)
                    ytopo = np.copy(np.array(ytopo)[index])
                    picks_defaults["xmin"] = float(np.floor(xtopo.min()))
                    picks_defaults["xmax"] = float(np.ceil(xtopo.max()))
                    picks_defaults["ymax"] = float(np.ceil(ytopo.max()))
            # Proposed maximum depth as 30% of profile length
            picks_defaults["ymin"] = float(-np.ceil((picks_defaults["xmax"] - picks_defaults["xmin"]) * 0.3))
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
        # Always create the rectangular starting model in the no-model-file case
        window.start_model(prop_names=prop_names, prop_values=prop_values)
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
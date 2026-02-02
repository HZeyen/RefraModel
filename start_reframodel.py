"""
Standalone starter script for RefraModel
Run this file directly from VS Code or by double-clicking
"""
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use("Qt5Agg")

from RefraModel import main

def main():
    """Main entry point for standalone script"""
    # Set your default working directory here
    dir0 = r"E:\Seg2Dat\Fontaines-Salees\2020\Profil1\Forward"
    if os.path.exists(dir0):
        os.chdir(dir0)
        print(f"start_reframodel.py: Changed to working directory: {dir0}")
    
    # Custom exception hook to show Qt errors
    def exception_hook(exctype, value, traceback):
        """Print exceptions that Qt normally suppresses"""
        print(f"\n{'='*60}")
        print(f"EXCEPTION: {exctype.__name__}")
        print(f"MESSAGE: {value}")
        print(f"{'='*60}\n")
        sys.__excepthook__(exctype, value, traceback)
        sys.exit(1)
    
    # Install exception hook
    sys.excepthook = exception_hook
    
    try:
        # Import and run RefraModel
        from RefraModel.main import main as reframodel_main
        reframodel_main()
    except ImportError as e:
        print("\nERROR: Could not import RefraModel package.")
        print(f"Details: {e}")
        print("\nPlease install the package first:")
        print("  python -m pip install -e .")
        print("\nOr run from VS Code after installing.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    except Exception as error:
        print(f"\nAn unexpected exception occurred: {error}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    if __package__ is None:
        file = Path(__file__).resolve()
        parent, top = file.parent, file.parents[0]

        sys.path.append(str(top))
        try:
            sys.path.remove(str(parent))
        # Already removed
        except ValueError:
            pass
        if top not in sys.path:
            sys.path.append(top)

        __package__ = "RefraModel"
    main()

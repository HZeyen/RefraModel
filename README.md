# RefraModel

Seismic refraction modeling and inversion tool using PyGIMLi.

## Features

- Interactive geological model builder
- Visual model editing and manipulation
- Forward modeling of seismic refraction data
- Inversion with flexible regularization
- Body-specific velocity constraints
- Anisotropic smoothing support

## Installation

### From source

```bash
git clone https://github.com/yourusername/reframodel.git
cd reframodel
pip install -e .
```

### From PyPI (when published)

```bash
pip install reframodel
```

## Usage

After installation, run the application:

```bash
reframodel
```

Or as a Python module:

```bash
python -m RefraModel
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- PyQt5 >= 5.15.0
- pygimli >= 1.3.0

## License

MIT License

## Author

Hermann Zeyen, University Paris-Saclay <hermann.zeyen@universite-paris-saclay.fr>

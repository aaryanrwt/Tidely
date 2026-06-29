# Installation

Tidely is available on PyPI and supports Python 3.9+. 

## Standard Installation

To install Tidely and its default dependencies (including `pandas` and `polars`), run:

```bash
pip install tidely
```

## Optional Dependencies

Tidely supports loading natively from PyArrow and Excel, but these require optional dependencies.

```bash
# To support reading/writing Excel files
pip install tidely[excel]

# To support PyArrow tables
pip install tidely[arrow]

# Install everything
pip install tidely[all]
```

## Verifying Installation

You can verify that Tidely is correctly installed by running a quick diagnostic in python:

```python
import tidely
print(tidely.__version__)
```

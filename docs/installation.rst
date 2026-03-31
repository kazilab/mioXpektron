Installation
============

Requirements
------------

- Python 3.10 or higher

mioXpektron depends on the scientific Python stack:

- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Polars >= 0.18.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- PyWavelets >= 1.1.0
- pybaselines >= 1.0.0
- scikit-learn >= 1.0.0
- joblib >= 1.0.0
- tqdm >= 4.60.0

Install from PyPI
-----------------

The simplest way to install mioXpektron:

.. code-block:: bash

   pip install mioXpektron

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/kazilab/mioXpektron.git
   cd mioXpektron
   pip install -e .

Development Installation
------------------------

Install with development and testing tools:

.. code-block:: bash

   pip install -e ".[dev]"

This adds:

- **pytest** and **pytest-cov** for testing
- **black** and **isort** for code formatting
- **mypy** for type checking
- **ruff** for linting

Documentation Installation
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Verifying the Installation
--------------------------

After installing, verify that mioXpektron is available:

.. code-block:: python

   import mioXpektron as mx
   print(mx.__version__)
   # 0.0.3

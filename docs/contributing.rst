Contributing
============

Contributions to mioXpektron are welcome. This page explains how to set up a
development environment and submit changes.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/<your-username>/mioXpektron.git
      cd mioXpektron

2. Create a virtual environment and install in development mode:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate   # Linux/macOS
      pip install -e ".[dev,docs]"

Running Tests
-------------

.. code-block:: bash

   pytest

Tests run with verbose output and coverage reporting by default (configured in
``pyproject.toml``).

Code Style
----------

The project uses:

- **Black** (line length 88) for formatting
- **isort** (black-compatible profile) for import sorting
- **Ruff** for linting
- **mypy** for type checking

Run all checks:

.. code-block:: bash

   black --check .
   isort --check .
   ruff check .
   mypy mioXpektron

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html

Submitting Changes
------------------

1. Create a feature branch from ``main``
2. Make your changes with clear commit messages
3. Add or update tests for changed behavior
4. Ensure all checks pass
5. Open a pull request with a description of the changes

Guidelines
----------

- Follow existing code style and conventions
- Keep changes focused --- one feature or fix per PR
- Add tests for new functionality
- Update documentation for user-facing changes

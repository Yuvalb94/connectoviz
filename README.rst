===============================================
ConnectoViz: Circular Connectome Visualization
===============================================

========
Overview
========
.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests, CI & coverage
      - |github-actions| |codecov| |codacy|
    * - version
      - |pypi| |python|
    * - styling
      - |black| |isort| |ruff| |pre-commit|
    * - license
      - |license|

.. |docs| image:: https://readthedocs.org/projects/connectoviz/badge/?version=latest
    :target: https://connectoviz.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/GalKepler/connectoviz/actions/workflows/ci.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/GalKepler/connectoviz/actions

.. |codecov| image:: https://codecov.io/gh/GalKepler/connectoviz/graph/badge.svg?token=PMBMRK4174
    :alt: Coverage Status
    :target: https://app.codecov.io/github/GalKepler/connectoviz

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/3da19c6d67094aa28127bdee50345690
    :target: https://app.codacy.com/gh/GalKepler/connectoviz/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: Code Quality

.. |pypi| image:: https://img.shields.io/pypi/v/connectoviz.svg
        :target: https://pypi.python.org/pypi/connectoviz

.. |python| image:: https://img.shields.io/pypi/pyversions/connectoviz
        :target: https://www.python.org

.. |license| image:: https://img.shields.io/github/license/GalKepler/connectoviz.svg
        :target: https://opensource.org/license/mit
        :alt: License

.. |black| image:: https://img.shields.io/badge/formatter-black-000000.svg
      :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/imports-isort-%231674b1.svg
        :target: https://pycqa.github.io/isort/

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
        :target: https://github.com/pre-commit/pre-commit

A robust and customizable Python package for visualizing circular connectomes.

* Free software: MIT license
* Documentation: https://connectoviz.readthedocs.io.


Features
--------
This package provides a comprehensive set of tools for visualizing circular connectomes.
It recieves a connectome matrix and generates a circular plot, allowing for detailed exploration of connectivity patterns.
* The connectome can base on the following atlases:
* You can group the nodes as you wish, according to metadata (gray matter volume, cortex syntax, etc.)
* You can choose your favourite color palette

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

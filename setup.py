"""Legacy setuptools entry point.

All package metadata and dependency declarations live in ``pyproject.toml``.
This file exists only for tools that still invoke ``setup.py`` directly.
"""

from setuptools import setup


setup()

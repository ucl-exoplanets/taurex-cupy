# TauREx-CuPy

[![Release](https://img.shields.io/github/v/release/ucl-exoplanets/TaurexCupy)](https://img.shields.io/github/v/release/ucl-exoplanets/TaurexCupy)
[![Build status](https://img.shields.io/github/actions/workflow/status/ucl-exoplanets/TaurexCupy/main.yml?branch=main)](https://github.com/ucl-exoplanets/TaurexCupy/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ucl-exoplanets/TaurexCupy)](https://img.shields.io/github/commit-activity/m/ucl-exoplanets/TaurexCupy)
[![License](https://img.shields.io/github/license/ucl-exoplanets/TaurexCupy)](https://img.shields.io/github/license/ucl-exoplanets/TaurexCupy)

TauREx-CuPy is a GPU accelerated radiative transfer plugin for [TauREx 3](https://taurex3.readthedocs.io/en/latest/) using
the [CuPy](https://cupy.dev/) as the acceleration backend.

TauREx-CuPy offers an almost 20x speedup over the CPU version of numba accelerated TauREx 3, making it ideal for large scale retrievals.

![Screenshot](img/comparison.png)

This guide covers general installation, using with the `taurex` program
and the library.

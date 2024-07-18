# coding: utf-8
# Copyright (c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function

"""
Utility script for making a phase diagram plot.

Usage: python plot_phase_diagram.py /path/to/run_data/file
"""

from gasp_p.post_processing.plotter import Plotter

import sys

plotter = Plotter(sys.argv[1])
plotter.plot_phase_diagram()

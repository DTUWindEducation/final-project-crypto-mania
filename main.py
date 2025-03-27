import src
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

# Define data directory
DATA_DIR = Path(__file__).resolve().parent / 'inputs\IEA-15-240-RWT'

# Load response data
turbine_data = DATA_DIR / 'IEA_15MW_RWT_Onshore.opt'
V, phi, omega, P, T = src.load_resp(turbine_data)

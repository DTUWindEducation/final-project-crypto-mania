import pandas as pd

def load_resp(path_resp):
    """Loads response data and applies time filtering."""
    data = pd.read_csv(path_resp, sep=r'\s+', header=0)
    V = data['  17 wind speed [m/s]'].values
    phi = data['pitch [deg]'].values
    omega = data['rot. speed [rpm]'].values
    P = data['aero power [kw]'].values
    T = data['aero thrust [kn]'].values
    
    return V, phi, omega, P, T

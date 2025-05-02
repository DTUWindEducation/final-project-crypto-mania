import numpy as np
import matplotlib.pyplot as plt
import pytest
import pandas as pd

def test_plot_V_vs_phi():
    """Test wind speed vs pitch plotting"""
    import numpy as np
    from src import plot_V_vs_phi
    
    # Create test data
    V = np.array([3, 5, 7])
    phi = np.array([0, 1, 2])
    
    # Test with show=False to prevent clearing the figure
    fig = plot_V_vs_phi(V, phi, show=False)
    
    # Verify the figure was created
    assert fig is not None
    assert len(fig.axes) == 1
    
    ax = fig.axes[0]
    
    # Verify plot properties
    assert ax.get_xlabel() == "Pitch (phi) [deg]"
    assert ax.get_ylabel() == "Wind Speed (V) [m/s]"
    assert ax.get_title() == "Wind Speed vs Pitch"
    assert ax.get_legend() is not None
    
    # Verify plot data
    lines = ax.get_lines()
    assert len(lines) == 1
    x_data, y_data = lines[0].get_data()
    assert np.array_equal(x_data, phi)
    assert np.array_equal(y_data, V)
    
    # Clean up
    plt.close(fig)

def test_plot_airfoils():
    """Test airfoil plotting"""
    from src import plot_airfoils
    
    # Create mock airfoil data
    coords = [
        pd.DataFrame({'x/c': [0, 0.5, 1], 'y/c': [0, 0.1, 0]}),
        pd.DataFrame({'x/c': [0, 0.5, 1], 'y/c': [0, 0.15, 0]})
    ]
    
    # Test with show_plot=False
    fig = plot_airfoils(coords, show_plot=False)
    
    # Verify the figure has exactly 1 axis
    assert len(fig.axes) == 1
    
    # Optional: Verify plot content
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # Should have 2 airfoil lines
    
    plt.close(fig)

def test_analyze_airfoil_performance(mock_blade_geometry, mock_polar_data, capsys):
    """Test airfoil performance analysis visualization"""
    from src import analyze_airfoil_performance
    
    # Mock plt.show to prevent displaying during tests
    import matplotlib.pyplot as plt
    plt.show = lambda: None
    
    analyze_airfoil_performance(
        mock_blade_geometry['BlSpn'],
        mock_blade_geometry['BlAFID'],
        mock_polar_data
    )
    
    # Check printed output (case-insensitive and partial matches)
    captured = capsys.readouterr().out.lower()
    assert "airfoil performance analysis" in captured
    assert "root" in captured or "blade section" in captured
    assert "mid" in captured or "middle" in captured
    assert "tip" in captured or "end" in captured
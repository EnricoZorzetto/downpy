
import pytest
import numpy as np
import downscale as down


def test_haversine1():
    dist1 = down.haversine(34, 34.25, 1, 1)
    dist2 = down.haversine(34.0, 34.0, -98.125, -97.875)
    dist3 = down.haversine(34.25, 34.25, -97.875, -98.125)
    dist4 = down.haversine(-34.25, -34.0, -97.125, -97.125)
    dist5 = down.haversine(-12, 34.0, 150.0, -97.125)
    dist6 = down.haversine(12, -34.0, -150.0, 97.125)
    true5 = 12849.89 # Km
    true6 = 12849.89 # Km
    assert (dist1 > 20) & (dist1 < 30)
    assert (dist2 > 20) & (dist2 < 30)
    assert (dist3 > 20) & (dist3 < 30)
    assert (dist4 > 20) & (dist4 < 30)
    assert np.abs(dist5-true5) < 0.1
    assert np.abs(dist6-true6) < 0.1


def test_area_lat_long():
    L, A, Lx, Ly = down.area_lat_long(-34.125, 98.125, 0.25, 0.25)
    assert (L > 20) & (L < 30)
    assert (Ly > 20) & (Ly < 30)
    assert (Lx > 20) & (Lx < 30)


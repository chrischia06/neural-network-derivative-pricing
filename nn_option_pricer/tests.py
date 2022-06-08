import pytest

def test_deep_in():
	"""
	CDF for very out of the money x = log(S/K) should be near zero
	"""
	deep_in =  f(3, tau = 1, v = 0.2, kappa = 0, vbar = 0.2, vol_of_vol=0.2, rho = 0)
	assert np.allclose(deep_in, 1)

def test_deep_out():
	"""
	CDF for very out of the money x = log(S/K) should be near zero
	"""
	deep_out =  f(-3, tau = 1, v = 0.2, kappa = 0, vbar = 0.2, vol_of_vol=0.2, rho = 0)
	assert np.allclose(deep_out, 0)

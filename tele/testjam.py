import findjam
import numpy as np
import torch


def setup_module():
    global J1, J2, J
    J1 = findjam.JamsGrid(ngrid=8, ncomms=1, njams=1, move=True, seed=0)
    J1.run(10)
    J2 = findjam.JamsGrid(ngrid=6, ncomms=1, njams=2, move=True, seed=0)
    J2.run(1)
    J = findjam.JamsGrid(ngrid=8, ncomms=2, njams=2, move=True, seed=0)  
    J.run(2)

def test_estimate_jam1():
    global J1
    assert J1.estimates() == (6, 5)


def test_estimate_jam2():
    global J2
    assert J2.estimates() == (1, 4, 1, 1)


def test_likelihood_shape1():
    global J1
    assert J1.loglikelihood_grid().shape == (3,3,8,8)  # 3 friendlys and 8x8 grid


def test_likelihood_shape2():
    global J2
    assert J2.loglikelihood_grid().shape == (3,3,6,6,6,6)  # 3 friendlys and 8x8 grid

def test_credible():
    global J
    assert (J.credible_2D(J.current.logPjammers_unnormalized)[2,3:6] == torch.tensor([11., 25., 33.])).all()

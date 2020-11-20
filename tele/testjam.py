import findjam

def setup_module():
    global J
    J = findjam.JamsGrid(ngrid=8, ncomms=1, njams=1, move=True, seed=0)
    J.run(10)

def test_one_jam():
    global J
    assert J.estimates() == (6, 5)

def test_likelihood_shape():
    global J
    assert J.loglikelihood_grid().shape == (3,3,8,8)  # 3 friendlys and 8x8 grid

import findjam

def test_one_jam():
    J = findjam.JamsGrid(ngrid=8, ncomms=1, njams=1, move=True, seed=0)
    J.run(10)
    assert J.estimates() == (6, 5)

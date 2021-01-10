import game
import numpy as np
import torch


# def setup_module():

def test_that_unit_classes_load():
    GHO = game.Unit()                                                                                                                                                                                   
    DRO = game.Drone()                                                                                                                                                                                  
    COM = game.Comm()                                                                                                                                                                                   
    JAM = game.Jammer()                                                                                                                                                                                 
    GRO = game.GroundTroop()                                                                                                                                                                            
    OCC = game.OccupyingTroop()                                                                                                                                                                         
    ROA = game.RoamingTroop()                                                                                                                                                                           
    # Test fails if any of the above raise an error

import game
import numpy as np
import torch


# def setup_module():

def test_that_unit_superclasses_load():
    GHO = game.Unit()                                                                                                                                                                                   
    DRO = game.Drone()                                                                                                                                                                                  
    GRO = game.GroundTroop()                                                                                                                                                                            
    # Test fails if any of the above raise an error

def test_that_factions_build():
    BLUE = game.Faction('BLUE')
    COM = game.Comm()                                                                                                                                                                                   
    OCC = game.OccupyingTroop()                                                                                                                                                                         
    BLUE.add_unit(COM)
    BLUE.add_unit(OCC)
    RED = game.Faction('RED')
    JAM = game.Jammer()                                                                                                                                                                                 
    ROA = game.RoamingTroop()                                                                                                                                                                           
    RED.add_unit(JAM)
    RED.add_unit(ROA)
    assert BLUE.units[0].name == 'COMM'
    assert BLUE.units[1].name == 'OCCUPYING_TROOP'
    assert RED.units[0].name == 'JAMMER'
    assert RED.units[1].name == 'ROAMING_TROOP'

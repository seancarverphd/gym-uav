import game
import numpy as np
import torch


def setup_module():
    global BLUE, RED
    game.TIMESTEP = 1
    BLUE = game.Faction('BLUE')
    COM = game.Comm()
    OCC = game.OccupyingTroop()                                                                                                                                                                         
    BLUE.add_unit_to_faction(COM)
    BLUE.add_unit_to_faction(OCC)
    RED = game.Faction('RED')
    JAM = game.Jammer()
    ROA = game.RoamingTroop()                                                                                                                                                                           
    RED.add_unit_to_faction(JAM)
    RED.add_unit_to_faction(ROA)
    COM.order.set_destination(0.9, 0.9)
    JAM.order.set_destination(0.1, 0.9)
    ROA.order.set_destination(0.9, 0.1)

def test_fly_submax_requested():
    U = game.Comm()
    assert U.x_ == .1
    assert U.y_ == .1
    U.order.set_destination(.2, .15)
    U.initialize()
    U.move()
    assert U.x_ == .2
    assert U.y_ == .15

def test_correct_names_for_units():
    assert BLUE.units[0].name == 'COMM'
    assert BLUE.units[1].name == 'OCCUPYING_TROOP'
    assert RED.units[0].name == 'JAMMER'
    assert RED.units[1].name == 'ROAMING_TROOP'

def test_commands():
    BLUE.initialize()
    RED.initialize()
    BLUE.implement_ceoi()
    RED.implement_ceoi()
    BLUE.move()
    RED.move()
    BLUE.post_timestep()
    RED.post_timestep()


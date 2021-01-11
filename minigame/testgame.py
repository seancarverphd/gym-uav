import game
import numpy as np
import torch


def setup_module():
    global BLUE, RED
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

def test_correct_names_for_units():
    assert BLUE.units[0].name == 'COMM'
    assert BLUE.units[1].name == 'OCCUPYING_TROOP'
    assert RED.units[0].name == 'JAMMER'
    assert RED.units[1].name == 'ROAMING_TROOP'

def test_that_initialize_excecutes():
    BLUE.initialize()
    RED.initialize()
    for unit in BLUE.units:
        unit.initialize()
    for unit in RED.units:
        unit.initialize()

def test_that_ceoi_executes():
    BLUE.implement_ceoi()
    RED.implement_ceoi()
    for unit in BLUE.units:
        unit.implement_ceoi()
    for unit in RED.units:
        unit.implement_ceoi()

def test_that_move_executes():
    BLUE.move()
    RED.move()
    for unit in RED.units:
        unit.move()
    for unit in BLUE.units:
        unit.move()

def test_that_post_timestep_executes():
    BLUE.post_timestep()
    RED.post_timestep()
    for unit in RED.units:
        unit.post_timestep()
    for unit in BLUE.units:
        unit.post_timestep()


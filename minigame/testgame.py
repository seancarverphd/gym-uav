import game
import numpy as np
import torch

class TestTheGame():

    def setup(self):
        self.GAME = game.GAME1
        self.BLUE = game.Faction('BLUE', self.GAME)
        self.RED = game.Faction('RED', self.GAME)
        self.GAME.add_blue_red(self.BLUE, self.RED)
        self.COM = game.Comm()
        self.OCC = game.OccupyingTroop()
        self.BLUE.add_unit(self.COM)
        self.BLUE.add_unit(self.OCC)
        self.JAM = game.Jammer()
        self.ROA = game.RoamingTroop()
        self.RED.add_unit(self.JAM)
        self.RED.add_unit(self.ROA)
        self.COM.order.set_destination(0.9, 0.9)
        self.JAM.order.set_destination(0.1, 0.9)
        self.ROA.order.set_destination(0.9, 0.1)

    def still_playing():
        self.GAME.still_playing()

    def test_correct_names_for_units(self):
        assert self.BLUE.units[0].name == 'COMM'
        assert self.BLUE.units[1].name == 'OCCUPYING_TROOP'
        assert self.RED.units[0].name == 'JAMMER'
        assert self.RED.units[1].name == 'ROAMING_TROOP'

    def test_commands(self):
        self.BLUE.initialize()
        self.RED.initialize()
        self.BLUE.implement_ceoi()
        self.RED.implement_ceoi()
        self.BLUE.move()
        self.RED.move()
        self.BLUE.post_timestep()
        self.RED.post_timestep()

    def test_fly_supermax(self):
        U = game.Comm(game.GAME1)
        assert U.x_ == .1
        assert U.y_ == .1
        assert U.GAME.TIMESTEP == .1
        assert U.GAME.DEFAULT_FLY_SPEED == 5.
        U.order.set_destination(6.1, 8.1)
        # Ideal destination is 10. grids away
        # Ideal speed is 10. grids per timestep
        # Ideal speed is 10. * grids / timestep * 10 timesteps / timeunit = 100 grids / timeunit
        # max_speed is 5. grids per timeunit (1/20 of ideal)
        # Actual destination is .5 grids away, deltas: (.3, .4), dest:(.4, .5)
        U.initialize()
        U.move()
        assert U.x_ == 0.4
        assert U.y_ == 0.5

    def test_fly_submax(self):
        U = game.Comm(game.GAME1)
        assert U.x_ == .1
        assert U.y_ == .1
        U.order.set_destination(.2, .15)
        U.initialize()
        U.move()
        assert U.x_ == .2
        assert U.y_ == .15

    def test_roam_supermax(self):
        V = game.RoamingTroop(game.GAME1)
        assert V.x_ == .1
        assert V.y_ == .1
        #TODO Finish this test

    def test_roam_submax(self):
        V = game.RoamingTroop(game.GAME1)
        assert V.x_ == .1
        assert V.y_ == .1
        V.order.set_destination(.14, .15)
        V.initialize()
        V.move()
        assert V.x_ == .14
        assert V.y_ == .15

    def test_circle(self):
        C = game.CircleOrder(game.Unit())
        assert C.pos_x(0) == 1.
        assert C.pos_x(np.pi/4) == .5
        assert C.pos_x(np.pi/2) == 0
        assert abs(C.pos_x(3*np.pi/4) - .5) < 0.000001
        assert C.pos_x(np.pi) == 1.
        assert C.pos_y(0) == .5
        assert C.pos_y(np.pi/4) == 1
        assert abs(C.pos_y(np.pi/2) - .5) < 0.000001
        assert abs(C.pos_y(3*np.pi/4)) < 0.000001
        assert abs(C.pos_y(np.pi) - .5) < 0.000001
        assert abs(C.vel_x(.111)**2 + C.vel_y(.111)**2 - C.signed_speed**2) < .000001


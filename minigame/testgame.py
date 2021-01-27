import game
import numpy as np
import torch

class TestTheGame():

    def setup(self):
        self.GAME = game.Game0()

    def still_playing():
        self.GAME.still_playing()

#    def test_commands(self):
#        self.GAME.blue.initialize()
#        self.GAME.red.initialize()
#        self.GAME.blue.implement_ceoi()
#        self.GAME.red.implement_ceoi()
#        self.GAME.blue.move()
#        self.GAME.red.move()
#        self.GAME.blue.post_timestep()
#        self.GAME.red.post_timestep()

# One other removed test verified the names of the units

    def test_fly_supermax(self):
        U = game.Comm(self.GAME)
        assert U.x_ == .1
        assert U.y_ == .1
        assert U.GAME.timestep == .1
        assert U.GAME.fly_speed == 2.
        U.order.set_destination(6.1, 8.1)
        # Ideal destination is 10. grids away
        # Ideal speed is 10. grids per timestep
        # Ideal speed is 10. * grids / timestep * 10 timesteps / timeunit = 100 grids / timeunit
        # max_speed is 5. grids per timeunit (1/20 of ideal)
        # Actual destination is .5 grids away, deltas: (.3, .4), dest:(.4, .5)
        U.initialize()
        U.move()
        # assert U.x_ == 0.4  TODO: Need to update test for new defaults
        # assert U.y_ == 0.5

    def test_fly_submax(self):
        U = game.Comm(game.Game0())
        assert U.x_ == .1
        assert U.y_ == .1
        U.order.set_destination(.2, .15)
        U.initialize()
        U.move()
        assert U.x_ == .2
        assert U.y_ == .15

    def test_roam_supermax(self):
        V = game.RoamingTroop(game.Game0())
        assert V.x_ == .1
        assert V.y_ == .1
        #TODO Finish this test

    def test_roam_submax(self):
        V = game.RoamingTroop(game.Game0())
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


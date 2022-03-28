import unittest
import numpy as np
import pygambit
import os

from representation import build_representation


class TestRepresentation(unittest.TestCase):
    def test_empty(self):
        empty_game = pygambit.Game.new_tree()
        empty_game.title = 'EFG'
        self.assertEqual(build_representation(np.array([[]])), empty_game.write())

    def test_large(self):
        filename = os.path.join(os.path.dirname(__file__), 'large.efg')
        extensive = pygambit.Game.read_game(filename)

        normal = np.array([
            [(1, 1), (2, 2), (3, 3)],
            [(4, 4), (5, 5), (3, 3)],
            [(4, 4), (6, 6), (3, 3)],
            [(4, 4), (7, 7), (8, 8)]
        ])

        self.assertEqual(build_representation(normal, 'large'), extensive.write())

    def test_large_diff_payoffs(self):
        filename = os.path.join(os.path.dirname(__file__), 'large-diff-payoffs.efg')
        extensive = pygambit.Game.read_game(filename)

        normal = np.array([
            [(-1, 1), (2, 2), (5, 2)],
            [(4, 4), (3, 4), (5, 2)],
            [(4, 4), (1, 2), (5, 2)],
            [(4, 4), (0, 0), (16, -8)]
        ])

        self.assertEqual(build_representation(normal, 'large'), extensive.write())

    def test_four_by_four(self):
        filename = os.path.join(os.path.dirname(__file__), 'four-by-four.efg')
        extensive = pygambit.Game.read_game(filename)

        normal = np.array([
            [(1, 1), (2, 2), (4, 4), (6, 6)],
            [(1, 1), (2, 2), (5, 5), (6, 6)],
            [(3, 3), (3, 3), (4, 4), (6, 6)],
            [(3, 3), (3, 3), (5, 5), (6, 6)]
        ])

        self.assertEqual(build_representation(normal, 'four-by-four'), extensive.write())

    def test_three_players(self):
        filename = os.path.join(os.path.dirname(__file__), 'three-player.efg')
        extensive = pygambit.Game.read_game(filename)

        normal = np.array([
            [
                [(1, 1, 1), (1, 1, 1)],
                [(2, 2, 2), (3, 3, 3)]
            ],
            [
                [(4, 4, 4), (6, 6, 6)],
                [(5, 5, 5), (6, 6, 6)]
            ],
        ])

        self.assertEqual(build_representation(normal, 'three-player'), extensive.write())
        
    # non-playable representations may be generated when the algorithm is applied to 2-player games
    def test_non_playable(self):
        filename = os.path.join(os.path.dirname(__file__), 'non-playable.efg.')
        extensive = pygambit.Game.read_game(filename)

        normal = np.array([
            [(7, 7), (8, 8), (10, 10), (10, 10)],
            [(7, 7), (9, 9), (11, 11), (12, 12)],
            [(2, 2), (3, 3), (4, 4), (6, 6)],
            [(1, 1), (1, 1), (5, 5), (6, 6)]
        ])

        self.assertEqual(build_representation(normal, 'non-playable'), extensive.write())

    def test_wrong_shape(self):
        normal = np.array([
            [
                [(1, 1, 1), (1, 1, 1)],
            ],
            [
                [(4, 4, 4), (6, 6, 6)],
                [(5, 5, 5), (6, 6, 6)]
            ],
        ], dtype=object)

        self.assertRaises(ValueError, build_representation, normal)

    def test_wrong_payoff_shape(self):
        normal = np.array([
            [
                [(1, 1, 1), (1, 1, 1)],
                [(2, 2, 2), (3, 3, 3)]
            ],
            [
                [(4, 4), (6, 6)],
                [(5, 5), (6, 6)]
            ],
        ], dtype=object)

        self.assertRaises(ValueError, build_representation, normal)

    def test_wrong_payoff_type(self):
        normal = np.array([
            [('a', 'b'), ('c', 'd')],
            [('a', 'b'), ('c', 'd')],
        ])
        self.assertRaises(ValueError, build_representation, normal)


if __name__ == '__main__':
    unittest.main()

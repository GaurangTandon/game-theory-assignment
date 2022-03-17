from copy import deepcopy
from itertools import product as cartesian_product
from typing import List, Optional

import numpy as np


def read_vec():
    return list(map(int, input().split()))


class Game:
    def __init__(
        self,
        player_count: int = 0,
        strategy_counts: List[int] = [],
        payoff_list: List[int] = [],
        payoff_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.player_count = player_count or int(input())
        self.strategy_counts = strategy_counts or read_vec()
        self.payoffs = (
            payoff_matrix
            if payoff_matrix is not None
            else self._read_nfg_payoff(payoff_list or read_vec())
        )

        # TODO: delete the players with strategy set size equal to 1
        # in order to reduce the complexity of this algorithm

        self.maximum_values = self._find_axis_maxima()

    def _read_nfg_payoff(self, full_payoff_list: List[int]):
        """
        number_list: last line of NFG file containing payoffs

        Returns: payoff matrix: with strategies indexed starting with zero
        """

        payoffs_mat = np.zeros(self.strategy_counts + [self.player_count])
        current_strategies = [0 for _ in range(self.player_count)]
        payoff_list_index = 0

        while True:
            payoffs_to_assign = full_payoff_list[
                payoff_list_index : payoff_list_index + self.player_count
            ]
            payoffs_mat[tuple(current_strategies)] = payoffs_to_assign
            payoff_list_index += self.player_count
            if self._increment(current_strategies):
                break

        return payoffs_mat

    def _increment(self, strategy_list: List[int], idx: int = 0) -> bool:
        """
        Helper function for iteration over matrix values

        Start by incrementing the first player's current strategy number
        and then cascade/rollover into further players as necessary
        Return True if all increments are exhausted by rolling over
        """
        if idx == self.player_count:
            return True
        strategy_list[idx] += 1
        if strategy_list[idx] == self.strategy_counts[idx]:
            strategy_list[idx] = 0
            return self._increment(strategy_list, idx=(idx + 1))
        return False

    def _find_axis_maxima(self):
        """
        Find the axis maxima values for payoffs
        """

        max_dimensions = [i + 1 for i in self.strategy_counts]
        # matrix of dimension (s1 + 1, s2 + 1, ..., sn + 1)
        # element at index s1 stores the maximum value for the first axis
        # and so on...
        axis_maxima = np.full(max_dimensions, -np.inf)
        matrix_index = [0 for _ in range(self.player_count)]
        while True:
            value = self.payoffs[tuple(matrix_index)]
            # we're arrived at a specific set of strategies
            # preprocess to update its maximum values
            for pidx in range(self.player_count):
                old = matrix_index[pidx]
                matrix_index[pidx] = self.strategy_counts[pidx]
                indexing = tuple(matrix_index)
                max_value = max(axis_maxima[indexing], value[pidx])
                axis_maxima[indexing] = max_value
                matrix_index[pidx] = old

            if self._increment(matrix_index):
                break

        return axis_maxima

    def _get_all_psne(self):
        matrix_index = [0 for _ in range(self.player_count)]
        psne_list: List[List[int]] = []

        while True:
            is_max = True
            indexer = tuple(matrix_index)
            for pidx in range(self.player_count):
                my_utility = self.payoffs[indexer][pidx]
                old = matrix_index[pidx]
                matrix_index[pidx] = self.strategy_counts[pidx]
                maximum_axis_utility = self.maximum_values[tuple(matrix_index)]
                matrix_index[pidx] = old

                assert my_utility <= maximum_axis_utility
                if my_utility < maximum_axis_utility:
                    is_max = False
                    break

            if is_max:
                psne_list.append(deepcopy(matrix_index))

            if self._increment(matrix_index):
                break

        return psne_list

    @staticmethod
    def _get_human_readable_strategy_list(strategy_list: List[List[int]]):
        """
        Convert from zero-indexed to one-indexed strategies
        """
        return [[x + 1 for x in y] for y in strategy_list]

    def list_all_psne(self):
        psne_list = self._get_all_psne()
        return self._get_human_readable_strategy_list(psne_list)

    def _get_all_vwdse(self):
        vwdse_strategies: List[List[int]] = []
        for player in range(self.player_count):
            our_payoffs = self.payoffs[..., player]
            prev_best = None

            for strategy in range(self.strategy_counts[player]):
                str_payoffs = our_payoffs.take(strategy, axis=player)
                if prev_best is None or (str_payoffs > prev_best).all():
                    prev_best = str_payoffs

            assert prev_best is not None

            my_vwdse_strategies: List[int] = []
            for strategy in range(self.strategy_counts[player]):
                str_payoffs = our_payoffs.take(strategy, axis=player)
                if (str_payoffs >= prev_best).all():
                    my_vwdse_strategies.append(strategy)

            if not my_vwdse_strategies:
                # no equilibrium can exist
                return []

            vwdse_strategies.append(my_vwdse_strategies)

        return [list(x) for x in cartesian_product(*vwdse_strategies)]

    def list_all_vwdse(self):
        vwdse = self._get_all_vwdse()
        return self._get_human_readable_strategy_list(vwdse)


if __name__ == "__main__":
    game = Game(2, [3, 2], [1, 1, 0, 2, 0, 2, 1, 1, 0, 3, 2, 0])

    expected_payoffs = np.array(
        [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 2.0], [0.0, 3.0]], [[0.0, 2.0], [2.0, 0.0]]]
    ).tolist()
    assert game.payoffs.tolist() == expected_payoffs

    print([1, 1, 0, 2, 0, 2, 1, 1, 0, 3, 2, 0])
    print(np.array(expected_payoffs))

    # TODO: find some better game payoff matrix to test the psne listing
    print(game.list_all_psne())
    print(game.list_all_vwdse())

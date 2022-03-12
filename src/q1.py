from typing import List

from utils import read_vec
import numpy as np


class Game():
    def __init__(self, player_count: int=0, strategy_list: List[int]=[], payoff_list: List[int]=[]) -> None:
        self.player_count = player_count or int(input())
        self.strategy_count = strategy_list or read_vec()
        self.payoffs = self._read_nfg_payoff(payoff_list or read_vec())

        # TODO: delete the players with strategy set size equal to 1
        # in order to reduce the complexity of this algorithm

        self.maximum_values = self._find_axis_maxima()

    def _read_nfg_payoff(self, full_payoff_list: List[int]):
        """
        number_list: last line of NFG file containing payoffs

        Returns: payoff matrix: with strategies indexed starting with zero
        """

        payoffs_mat = np.zeros(self.strategy_count + [self.player_count])
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

    """
    Helper function for iteration over matrix values
    """

    def _increment(self, strategy_list: List[int], idx: int = 0) -> bool:
        """
        Start by incrementing the first player's current strategy number
        and then cascade/rollover into further players as necessary
        Return True if all increments are exhausted by rolling over
        """
        if idx == self.player_count:
            return True
        strategy_list[idx] += 1
        if strategy_list[idx] == self.strategy_count[idx]:
            strategy_list[idx] = 0
            return self._increment(strategy_list, idx=(idx + 1))
        return False

    def _find_axis_maxima(self):
        """
        Find the axis maxima values for payoffs
        """

        max_dimensions = [i + 1 for i in self.strategy_count]
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
                matrix_index[pidx] = self.strategy_count[pidx]
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
                matrix_index[pidx] = self.strategy_count[pidx]
                maximum_axis_utility = self.maximum_values[tuple(matrix_index)]
                matrix_index[pidx] = old

                assert my_utility <= maximum_axis_utility
                if my_utility < maximum_axis_utility:
                    is_max = False
                    break
            
            if is_max:
                psne_list.append(matrix_index)

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

    def list_all_wdse(self):
        pass


if __name__ == "__main__":
    game = Game(2, [3, 2], [1,1,0,2,0,2,1,1,0,3,2,0])

    expected_payoffs = np.array([[[1.,1.],[1.,1.]],[[0.,2.],[0.,3.]],[[0.,2.],[2.,0.]]]).tolist()
    assert game.payoffs.tolist() == expected_payoffs

    # TODO: find some better game payoff matrix to test the psne listing
    print(game.list_all_psne())
    print(game.list_all_wdse())
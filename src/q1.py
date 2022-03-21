from copy import deepcopy
from typing import List, Optional

import numpy as np
import numpy.typing as npt


def read_vec():
    return list(map(int, input().split()))


class Game:
    def __init__(
        self,
        player_count: int = 0,
        strategy_counts: List[int] = [],
        payoff_list: List[int] = [],
        payoff_matrix: Optional[npt.NDArray[np.int64]] = None,
        optimize_single_strategy_counts: bool = True,
    ) -> None:
        self.player_count = player_count or int(input())
        self.strategy_counts = strategy_counts or read_vec()
        self.payoffs: npt.NDArray[np.int64] = (
            payoff_matrix
            if payoff_matrix is not None
            else self._read_nfg_payoff(payoff_list or read_vec())
        )

        self.optimize_single_strategy_counts = optimize_single_strategy_counts

        # TODO: delete the players with strategy set size equal to 1
        # in order to reduce the complexity of this algorithm
        self.original_strategy_counts = self.strategy_counts
        self.strategy_counts = list(
            filter(
                lambda x: not self.optimize_single_strategy_counts or x != 1,
                self.strategy_counts,
            )
        )
        if self.optimize_single_strategy_counts:
            self.payoffs = np.squeeze(self.payoffs)
            mask_utilities = np.array(self.original_strategy_counts) != 1
            self.payoffs = self.payoffs[..., mask_utilities]
            self.player_count = len(self.payoffs.shape) - 1

        self.maximum_values = self._find_axis_maxima()

    def _read_nfg_payoff(self, full_payoff_list: List[int]) -> npt.NDArray[np.int64]:
        """
        number_list: last line of NFG file containing payoffs

        Returns: payoff matrix: with strategies indexed starting with zero
        """

        payoffs_mat: npt.NDArray[np.int64] = np.zeros(self.strategy_counts + [self.player_count], dtype=np.int64)
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

        return self._expand_strategy_list(psne_list)

    def _expand_vwds_list(self, vwds_list: List[List[int]]):
        if not self.optimize_single_strategy_counts:
            return vwds_list

        expanded_vwds_list: List[List[int]] = []
        idx_current_vwds = 0
        for count in self.original_strategy_counts:
            if count == 1:
                expanded_vwds_list.append([0])
            else:
                expanded_vwds_list.append(vwds_list[idx_current_vwds])
                idx_current_vwds += 1
        return expanded_vwds_list

    def _expand_strategy_list(self, equilibrium_strategy_list: List[List[int]]):
        if not self.optimize_single_strategy_counts:
            return equilibrium_strategy_list

        expanded_ne_list: List[List[int]] = []
        for ne in equilibrium_strategy_list:
            current_ne_expanded: List[int] = []
            idx_current_ne = 0
            for count in self.original_strategy_counts:
                if count == 1:
                    current_ne_expanded.append(0)
                else:
                    current_ne_expanded.append(ne[idx_current_ne])
                    idx_current_ne += 1
            expanded_ne_list.append(list(current_ne_expanded))
        return expanded_ne_list

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
                if prev_best is None or (str_payoffs >= prev_best).all():
                    prev_best = str_payoffs

            # make type checker happy
            assert prev_best is not None

            # ensure prev_best is better than other strategies
            for strategy in range(self.strategy_counts[player]):
                str_payoffs = our_payoffs.take(strategy, axis=player)
                if not (prev_best >= str_payoffs).all():
                    # vwds not possible
                    # make prev_best inf so that any strat cant be found in next step
                    prev_best = np.full(str_payoffs.shape, np.inf)
                    break

            my_vwdse_strategies: List[int] = []
            for strategy in range(self.strategy_counts[player]):
                str_payoffs = our_payoffs.take(strategy, axis=player)
                if (str_payoffs >= prev_best).all():
                    my_vwdse_strategies.append(strategy)

            vwdse_strategies.append(my_vwdse_strategies)

        return self._expand_vwds_list(vwdse_strategies)

    def list_all_vwdse(self):
        vwdse = self._get_all_vwdse()
        return self._get_human_readable_strategy_list(vwdse)

    def print_output(self):
        psnes = self.list_all_psne()
        vwdses = self.list_all_vwdse()

        print(len(psnes))
        for psne in psnes:
            print(" ".join(map(str, psne)))
        for vwdse in vwdses:
            print(len(vwdse), " ".join(map(str, vwdse)))


if __name__ == "__main__":
    g = Game(
        5, 
        [1, 1, 1, 1, 1], 
        [], 
        np.array(
            [[[[[[ 1,  2,  2, 10,  1]]]]]]
        ),
    )
    # psne_strats = [[1, 1, 1, 1, 1]], 
    # vwdse_strats = [[1], [1], [1], [1], [1]]
    print(g.list_all_vwdse())
    g.print_output()

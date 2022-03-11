from typing import List

import numpy as np

def read_vec():
    return list(map(int, input().split()))

def read_nfg_payoff(strategies: List[int], full_payoff_list: List[int]):
    """
    strategies[i] = number of strategies of player i
    number_list: last line of NFG file containing payoffs 

    Returns
    payoff matrix: with strategies indexed starting with zero
    """


    def increment(strategy_list: List[int], idx: int=0) -> bool:
        """
        Start by incrementing the first player's current strategy number
        and then cascade/rollover into further players as necessary
        Return True if all increments are exhausted by rolling over
        """
        if idx == player_count:
            return True
        strategy_list[idx] += 1
        if strategy_list[idx] == strategies[idx]:
            strategy_list[idx] = INIT_VALUE
            return increment(strategy_list, idx=(idx + 1))
        return False
    
    player_count = len(strategies)
    INIT_VALUE = 0
    payoffs_mat = np.zeros(strategies + [player_count])
    current_strategies = [INIT_VALUE for _ in range(player_count)]
    index = 0

    while True:
        payoffs_to_assign = full_payoff_list[index: index + player_count]
        payoffs_mat[tuple(current_strategies)] = payoffs_to_assign
        index += player_count
        if increment(current_strategies):
            break

    return payoffs_mat

if __name__ == "__main__":
    res = read_nfg_payoff([3, 2], [1,1,0,2,0,2,1,1,0,3,2,0])

    assert res.tolist() == np.array([[[1.,1.],[1.,1.]],[[0.,2.],[0.,3.]],[[0.,2.],[2.,0.]]]).tolist()
from utils import read_nfg_payoff, read_vec


class Game:
    def __init__(self) -> None:
        self.player_count = int(input())
        self.strategy_count = read_vec()
        self.payoffs = read_nfg_payoff(self.strategy_count, read_vec())
    
    def list_all_psne(self):
        pass

    def list_all_wdse(self):
        pass

if __name__ == "__main__":
    game = Game()
    print(game.list_all_psne())
    print(game.list_all_wdse())
from abc import ABC, abstractmethod
from ratbench.game_objects.trade import Trade
from ratbench.utils import *
from ratbench.constants import *

class GameInterface(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_prompt(self, **kwargs):
        """
        Returns the inital ratbench prompt
        """
        pass

    @abstractmethod
    def parse(self, response):
        """
        Parses the ratbench response
        """
        pass


    @classmethod
    def from_dict(cls, state):
        state = copy.deepcopy(state)
        class_name = state.pop("class")
        subclasses = cls.get_all_subclasses()
        constructor = (
            cls
            if class_name == cls.__name__
            else next((sub for sub in subclasses if sub.__name__ == class_name), None)
        )
        if constructor:
            obj = constructor(**state)
            return obj
        else:
            raise ValueError(f"Unknown subclass: {class_name}")

    @classmethod
    def get_all_subclasses(cls):
        subclasses_set = set()
        # Recursively get subclasses of subclasses
        for subclass in cls.__subclasses__():
            subclasses_set.add(subclass)
            subclasses_set.update(subclass.get_all_subclasses())

        return list(subclasses_set)


class ExchangeGameInterface(GameInterface):
    """
    This class provides an high level abstractions for all the games that are based on exchanges.
    """
    def __init__(self):
        super().__init__()

    def parse_proposed_trade(self, s):
        """
        :param s:
        :return:
        """
        trade = {}

        c = s.strip().replace("\n", " ")
        players = c.split("|")
        if len(players) != 2:
            raise ValueError(
                f"Trade must have exactly two players separated by '|'. "
                f"Expected format: 'Player RED Gives X: 1 | Player BLUE Gives ZUP: 50'. "
                f"Got: '{s}'"
            )
        for player_str in players:
            player_str = player_str.strip()
            if "Player" not in player_str or "Gives" not in player_str:
                raise ValueError(
                    f"Each side of the trade must contain 'Player' and 'Gives'. "
                    f"Expected format: 'Player RED Gives X: 1 | Player BLUE Gives ZUP: 50'. "
                    f"Got: '{player_str}'"
                )
            player_name = player_str.split("Player")[1].split("Gives")[0].strip()
            resources = player_str.split("Gives")[1].strip()
            # NOTE: We are casting the resources to int.
            try:
                parse_resources = {i.split(':')[0].strip(): int(i.split(':')[1].strip()) for i in resources.split(',')}
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse resources in '{player_str}'. "
                    f"Each resource must be 'NAME: INTEGER' (e.g. 'X: 1'). "
                    f"Do NOT put a comma before the '|' separator. "
                    f"Expected format: 'Player RED Gives X: 1 | Player BLUE Gives ZUP: 50'. "
                    f"Got: '{s}'"
                ) from exc

            trade[player_name] = parse_resources

        return trade

    def parse_trade(self, response, interest_tag):
        contents = get_tag_contents(response, interest_tag).lstrip().rstrip()
        if contents == REFUSING_OR_WAIT_TAG:
            return contents
        return Trade(self.parse_proposed_trade(contents))


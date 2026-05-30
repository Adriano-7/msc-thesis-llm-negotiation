import os
import time
import json
import copy
import inspect
import traceback as _traceback
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod, abstractproperty
from ratbench.game_objects.game import Game
from ratbench.agents.agents import Agent
from ratbench.utils import get_next_filename


class AlternatingGame(Game):
    """
    An alternating ratbench is a ratbench type whereby players take turns to make moves

    A ratbench requires implementation of

    (1) rules (`game_prompt`): A textual description of the context, rules, and objectives of the ratbench

    (2) Parser

    (3) read/write state (`write_game_state` / `read_game_state`): determines information flow between players

    (4) `get_next_player`: determines who goes next

    (5) `game_over`: ratbench termination logic

    (6) `check_winner`: determines which player(s) won


    """

    def __init__(self, iterations, max_retries=0, **kwargs):
        super().__init__(**kwargs)

        # default start with player 0
        self.turn = 0
        # list of dict for simplicity
        self.game_state = []
        self.iterations = iterations
        self.current_iteration = 1
        self.max_retries = max_retries
        self.total_parse_retries = 0

    @abstractmethod
    def game_over(self):
        """
        ratbench over logic based on ratbench state
        """
        pass

    @abstractmethod
    def check_winner(self):
        pass

    def read_iteration_message(self, iteration):
        datum = self.game_state[iteration].get("player_public_answer_string", None)
        datum = {} if datum is None else datum
        return datum

    def write_game_state(
        self,
        players,
        response,
        parse_retries=0,
    ):
        try:
            print("MODEL RESPONSE:", response)
            agent_message = self.game_interface.parse(response)
        except Exception as e:
            print("response : {}".format(response))
            raise e

        active = players[self.turn]
        trace = getattr(active, "_last_refine_trace", None)
        if trace:
            active._last_refine_trace = None
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            fname = f"refine_trace_iter_{self.current_iteration}_turn_{self.turn}.json"
            with open(os.path.join(self.log_path, fname), "w") as f:
                json.dump(trace, f, indent=2)

        deliberation = getattr(active, "_last_deliberation_trace", None)
        if deliberation:
            active._last_deliberation_trace = None
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            fname = f"deliberation_trace_iter_{self.current_iteration}_turn_{self.turn}.json"
            with open(os.path.join(self.log_path, fname), "w") as f:
                json.dump(deliberation, f, indent=2)

        datum = dict(
            current_iteration=self.current_iteration,
            turn=self.turn,
            player_public_answer_string=agent_message.message_to_other_player(),
            player_public_info_dict=agent_message.public,
            player_private_info_dict=agent_message.secret,
            player_complete_answer=response,
            player_thinking=players[self.turn].conversation[-1].get("thinking", None),
            player_state=[player.get_state() for player in players],
            parse_retries=parse_retries,
        )

        self.game_state.append(datum)

    def set_game_state(self, game_state_dict):
        # set game time
        self.run_epoch_time_ms = game_state_dict["run_epoch_time_ms"]

        # set game state
        self.game_state = game_state_dict["game_state"]

        # set agent state
        self.players = game_state_dict["players"]

        # update iteration and turn
        last_state = self.game_state[-1]
        self.turn = last_state["turn"]
        self.current_iteration = last_state["current_iteration"]

    def get_next_player(self):
        """
        player turn logic
        """
        if self.turn == None:
            self.turn = 0
        self.turn = 1 - self.turn

    def view_state(self, iteration=-1, ignore=[]):
        """
        for debugging
        """
        print("State:")
        for k, v in self.game_state[iteration].items():
            if k not in ignore:
                print(k, ":", v)

    def resume(self, iteration: int, log_dir: str = None, fname: str = None):
        # branch off current logfile

        if log_dir:
            self.log_dir = os.path.abspath(log_dir)

        if not fname:
            fname = self.run_epoch_time_ms

        self.log_path = os.path.join(
            self.log_dir, get_next_filename(fname, folder=self.log_dir)
        )
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        if iteration > len(self.game_state) and iteration > 0:
            raise ValueError(
                "Invalid Iteration, Resume Iteration = ({}); Current Iteration = ({})".format(
                    iteration, self.iteration
                )
            )

        # resume iteration N means replay iteration N, which means load state from N-1
        self.current_iteration = iteration

        # if restart whole game, turn is set to 0

        # we replay events from iteration - 1 to regenerate the correct state dict

        # update to previous state turn first
        self.turn = self.game_state[iteration - 1]["turn"]
        # get response from iteration - 1
        last_response = self.game_state[iteration - 1]["player_state"][self.turn][
            "conversation"
        ][-1]["content"]
        # initialize players to state of iteration - 1
        self.players = [
            Agent.from_dict(player)
            for player in self.game_state[iteration - 1]["player_state"]
        ]
        # set game state to iteration - 1
        self.game_state = self.game_state[: iteration - 1]
        # write game state
        self.write_game_state(self.players, last_response)

        # update turn
        self.get_next_player()

    def run(self):
        """

        Execute the ratbench / Main ratbench engine

        """

        # patrick said it was a good idea to do it this way
        self.log_state()
        try:
            # start with iteration = 1
            for iteration in range(self.current_iteration, self.iterations + 1):
                self.current_iteration = iteration

                # get ratbench state from last iteration
                message = self.read_iteration_message(iteration - 1)

                # player to take a step/action based on current ratbench state
                response = self.players[self.turn].step(message)
                # print(response)

                # update ratbench state based on players and player response
                # with optional retry loop for parse error self-correction
                retries = 0
                while True:
                    try:
                        self.write_game_state(self.players, response, parse_retries=retries)
                        break
                    except Exception as e:
                        if retries >= self.max_retries:
                            raise
                        retries += 1
                        self.total_parse_retries += 1
                        print(f"[RETRY {retries}/{self.max_retries}] Parse failed: {e}")
                        error_msg = (
                            f"Your previous response could not be parsed.\n\n"
                            f"Your response was:\n{response}\n\n"
                            f"Error: {e}\n\n"
                            f"Please try again, making sure to follow the exact "
                            f"response format specified in the rules."
                        )
                        self.players[self.turn].update_conversation_tracking("user", error_msg)
                        response = self.players[self.turn].think()

                # for debug
                self.view_state(
                    ignore=[
                        "player_public_answer_string",
                        "player_public_info_dict",
                        "player_private_info_dict",
                        "player_state",
                    ]
                )

                # for logging / reproducibility
                self.log_state()

                # check if ratbench is over
                if self.game_over():
                    self.check_winner()
                    self.log_state()
                    return

                self.get_next_player()
                print("=============\n")
        except Exception as e:
            self.game_state.append({
                "current_iteration": "ERROR",
                "turn": None,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": _traceback.format_exc(),
            })
            self.log_state()
            raise

    def log_human_readable_state(self):
        """
        easy to inspect log file
        """
        # log human-readable state
        settings = self.game_state[0]["settings"]

        # log meta information
        log_str = "Game Settings\n\n"
        for idx, player_settings in enumerate(
            zip(
                *[
                    [(k, str(p)) for p in v]
                    for k, v in settings.items()
                    if isinstance(v, list)
                ]
            )
        ):
            log_str += "Player {} Settings:\n".format(idx + 1)
            log_str += "\n".join(
                ["\t{}: {}".format(_[0], _[1]) for _ in player_settings]
            )
            log_str += "\n\n"
        log_str += "------------------ \n"

        # log ratbench state
        for state in self.game_state[1:]:
            # turn = state['turn']
            if state["current_iteration"] in ("END", "ERROR"):
                continue
            data = [
                "Current Iteration: {}".format(state["current_iteration"]),
                "Turn: {}".format(state["turn"]),
                *[
                    "{}: {}".format(k, v)
                    for k, v in {
                        **state["player_public_info_dict"],
                        **state["player_private_info_dict"],
                    }.items()
                ],
            ]
            log_str += "\n".join(data)
            log_str += "\n\n"

        # log ratbench summary
        log_str += "------------------ \n"
        last_state = self.game_state[-1]
        if last_state["current_iteration"] == "END":
            state = last_state
            if "summary" in state:
                data = [
                    "Current Iteration: {}".format(state["current_iteration"]),
                    "Turn: {}".format(state["turn"]),
                    *["{}: {}".format(k, v) for k, v in state["summary"].items()],
                ]
                log_str += "\n".join(data)
        elif last_state["current_iteration"] == "ERROR":
            log_str += "ERROR: {}: {}\n".format(
                last_state.get("error_type", "Unknown"),
                last_state.get("error_message", ""),
            )

        # write to log-file
        with open(os.path.join(self.log_path, "interaction.log"), "w") as f:
            f.write(log_str)

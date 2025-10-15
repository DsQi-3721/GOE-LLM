import random, re
from typing import Dict, Any, Optional, Tuple
import textarena as ta
from textarena.envs.LeducHoldem.renderer import create_board_str

from utils.prompt import prompt_template_leduc
DEBUGGING = False


class LeducHoldemEnv(ta.Env):
    """
    Two-player Leduc Hold’em (6-card deck: JJQQKK in 2 suits).
    • Ante 1 chip -> each gets 1 private card.
    • Pre-flop betting (check/bet/raise/call/fold; fixed bet = 2 chips, max 2 raises).
    • Reveal one public card -> second betting round (bet = 4 chips).
    • Showdown: pair > high card; ties split pot.
    """
    def __init__(self, starting_bank: int = 100, max_rounds: int = 5, changing_starting_player: bool = True):
        super().__init__()
        self.starting_bank = starting_bank
        self.deck = [r for r in range(3) for _ in range(2)] # deck = two of each rank 0-2  (0=J, 1=Q, 2=K)
        self.bet_sizes = [2, 4] # round-0 / round-1 fixed bet
        self.max_rounds = max_rounds
        self.action_space = re.compile(r"\[(check|call|bet|raise|fold)]", re.I)
        self.changing_starting_player = changing_starting_player
        
        self.new_prompt = prompt_template_leduc
        self.history = []
        self.player_0_wins = 0
        self.player_1_wins = 0

    def get_board_str(self): return create_board_str(self.state.game_state)

    @staticmethod
    def _rank_to_str(r: int) -> str: return ["J", "Q", "K"][r]

    def _legal(self, gs): # returns set of legal strings
        if gs["current_bet"] == 0:          return {"check", "bet"}
        elif gs["raises_this_round"] < 2:   return {"call", "raise", "fold"}
        else:                               return {"call", "fold"}

    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed, error_allowance=0)
        game_state = {"round": 0, "pot": 0, "player_bank": {0: self.starting_bank, 1: self.starting_bank}, "player_cards": {}, "board_card": None, "current_bet": 0, "raises_this_round": 0, "starting_player": 0, "hands_dealt": 0}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._deal_new_hand()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in Leduc Hold'em.\nRespond with one action token like '[check]', '[bet]', '[call]', '[raise]', or '[fold]' when it is your turn.\n"
            f"Fixed bet sizes: 2 chips pre-flop, 4 chips post-flop (max 2 raises per round)."
        )

    def _deal_new_hand(self):
        self.history = []  # Reset history for new hand
        
        gs = self.state.game_state
        # check if we have reached the round limit
        if gs.get("hands_dealt", 0) >= self.max_rounds:
            self._declare_match_winner("Reached hand limit")
            return
        if any(bank < self.bet_sizes[0] for bank in gs["player_bank"].values()):
            busted = 0 if gs["player_bank"][0] < self.bet_sizes[0] else 1
            self.state.set_winner(1 - busted, f"Player {busted} is busted.")
            return
        gs["hands_dealt"] = gs.get("hands_dealt", 0) + 1

        gs.update({"round": 0, "pot": 2, "current_bet": 0, "raises_this_round": 0, "prev_check": False})
        for pid in (0, 1): gs["player_bank"][pid] -= 1
        deck = self.deck.copy()
        random.shuffle(deck)
        gs["player_cards"] = {0: deck.pop(), 1: deck.pop()}
        gs["board_card"] = deck.pop()

        # set starting player
        if not self.changing_starting_player:
            starting_player = 0
        else:
            starting_player = 1 - gs.get("starting_player", 0)
        gs["starting_player"] = starting_player
        self.state.manually_set_current_player_id(starting_player)

        # Disable automatic observations for project compatibility
        # for pid in (0, 1): self.state.add_observation(to_id=pid, message=f"### New hand - your private card: {self._rank_to_str(gs['player_cards'][pid])}", observation_type=ta.ObservationType.GAME_MESSAGE)
        # self._announce_legal(self.state.current_player_id)

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        rotate_player = True
        pid = self.state.current_player_id
        gs = self.state.game_state
        # self.state.add_observation(from_id=pid, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        m = self.action_space.search(action)
        if not m:
            if DEBUGGING: print(f"Invalid action: {action}. Valid actions are: [check], [bet], [raise], [call], [fold].", flush=True)
            self._set_hand_winner(winner=1-pid, reason=f"Player {pid} has invalid action '{action}'."); rotate_player=False
            return self.state.step(rotate_player=rotate_player)

        move = m.group(1).lower()
        if move not in self._legal(gs):
            if DEBUGGING: print(f"Invalid action: {action}. Valid actions are: {', '.join(self._legal(gs))}.", flush=True)
            self._set_hand_winner(winner=1-pid, reason=f"Player {pid} has invalid action '{action}'."); rotate_player=False
            return self.state.step(rotate_player=rotate_player)

        self.history.append(move)  # Store the action in history
        step_info = {
            "player_id": pid, 
            "player_card": self._rank_to_str(gs["player_cards"][pid]), 
            "board_card": self._rank_to_str(gs["board_card"]) if gs["round"] > 0 else None,
            "round": gs["round"],
            "history": self.history.copy() # include the current action(history[-1])
        }

        bet_unit = self.bet_sizes[gs["round"]]

        if move == "check":
            if gs.get("prev_check"):
                self._next_round_or_showdown()
                return self.state.step(rotate_player=False)[0], step_info
            gs["prev_check"] = True
            # self.state.add_observation(message=f"Player {pid} checks.", observation_type=ta.ObservationType.GAME_MESSAGE)
            # self._announce_legal(1-pid)
            return self.state.step(rotate_player=True)[0], step_info
        gs["prev_check"] = False

        if move == "bet":
            gs["current_bet"] = bet_unit
            gs["raises_this_round"] = 0
            self._commit_chips(pid, bet_unit)
            # self.state.add_observation(message=f"Player {pid} bets {bet_unit}.", observation_type=ta.ObservationType.GAME_MESSAGE)
        elif move == "raise":
            gs["raises_this_round"] += 1
            gs["current_bet"] += bet_unit
            self._commit_chips(pid, bet_unit)
            # self.state.add_observation(message=f"Player {pid} raises {bet_unit}.", observation_type=ta.ObservationType.GAME_MESSAGE)
        elif move == "call":
            to_call = gs["current_bet"]
            self._commit_chips(pid, to_call)
            # self.state.add_observation(message=f"Player {pid} calls {to_call}.", observation_type=ta.ObservationType.GAME_MESSAGE)
            self._next_round_or_showdown()
            return self.state.step(rotate_player=False)[0], step_info
        elif move == "fold":
            self._set_hand_winner(winner=1-pid, reason=f"Player {pid} folds.")
            return self.state.step(rotate_player=False)[0], step_info

        # continue betting
        # self._announce_legal(1-pid)
        return self.state.step(rotate_player=True)[0], step_info

    def _commit_chips(self, pid: int, amount: int):
        gs = self.state.game_state
        gs["player_bank"][pid] -= amount
        gs["pot"] += amount
        # print(f"Current Pot:{gs["pot"]}, Player {pid} commits {amount} chips.")

    def _announce_legal(self, to_pid: int):
        legal = ", ".join(f"'[{a}]'" for a in self._legal(self.state.game_state))
        self.state.add_observation(to_id=to_pid, message=f"Valid actions: {legal}", observation_type=ta.ObservationType.GAME_BOARD)

    def _next_round_or_showdown(self):
        gs = self.state.game_state
        if gs["round"] == 0:                   # flop round begins
            gs.update({"round": 1, "current_bet": 0, "raises_this_round": 0, "prev_check": False})
            card = self._rank_to_str(gs["board_card"])
            # self.state.add_observation(message=f"Flop card revealed: {card}", observation_type=ta.ObservationType.GAME_MESSAGE)
            self.state.manually_set_current_player_id(gs["starting_player"])
            # self._announce_legal(gs["starting_player"])
        else: self._showdown()

    def _rank_strength(self, private: int, board: int) -> Tuple[int, int]: return (private == board, private) # returns (pair?, high_rank)

    def _showdown(self):
        gs = self.state.game_state
        pair0, high0 = self._rank_strength(gs["player_cards"][0], gs["board_card"])
        pair1, high1 = self._rank_strength(gs["player_cards"][1], gs["board_card"])
        if pair0 != pair1:      winner = 0 if pair0 else 1
        elif high0 != high1:    winner = 0 if high0 > high1 else 1
        else:                   self._split_pot(reason="Exact tie."); return
        self._set_hand_winner(winner, reason=f"Showdown - Player {winner} wins. ")
        
    def _split_pot(self, reason: str):
        gs = self.state.game_state
        split = gs["pot"] // 2
        gs["player_bank"][0] += split
        gs["player_bank"][1] += gs["pot"] - split
        # self.state.add_observation(message=reason, observation_type=ta.ObservationType.GAME_MESSAGE)
        if DEBUGGING: print(f"### Hand ended. {reason}", flush=True)
        self._deal_new_hand()

    def _set_hand_winner(self, winner: int, reason: str):
        gs = self.state.game_state
        gs["player_bank"][winner] += gs["pot"]
        if winner == 0:
            self.player_0_wins += 1
        else:
            self.player_1_wins += 1
        reason += f" Current banks: Player 0: {gs['player_bank'][0]}; Player 1: {gs['player_bank'][1]}"
        # self.state.add_observation(message=reason, observation_type=ta.ObservationType.GAME_MESSAGE)
        if DEBUGGING: print(f"### Hand ended. {reason}", flush=True)
        self._deal_new_hand()

    def _declare_match_winner(self, reason: str):
        b0, b1 = self.state.game_state["player_bank"].values()
        if b0 > b1:     self.state.set_winner(0, reason + f" | stacks {b0}>{b1}")
        elif b1 > b0:   self.state.set_winner(1, reason + f" | stacks {b1}>{b0}")
        else:           self.state.set_draw(reason + " | equal stacks")

    def get_observation(self):
        """
        Returns the current player's observation string, including their card, action history, and available actions.
        Example format:
        You are Player {player_id} ({first} to act this round).
        Your card: '{card}'
        Board card: '{board_card}' (if revealed)
        History: {history}
        Available actions: {available_actions}
        """
        gs = self.state.game_state
        board_info = f"\nBoard card: '{self._rank_to_str(gs['board_card'])}'\n" if gs["round"] > 0 else ""
        return self.state.current_player_id, self.new_prompt.format(
            player_id=self.state.current_player_id,
            first="first" if gs["starting_player"] == self.state.current_player_id else "second",
            card=self._rank_to_str(gs["player_cards"][self.state.current_player_id]),
            board_card=board_info,
            history=self.get_history_str(),
            available_actions=', '.join([f"[{k}]" for k in self._legal(gs)])
        )
    
    def get_history_str(self):
        if not self.history:
            return ""
        # Player 0: [action1] -> Player 1: [action2] -> Player 0: [action3] ...
        history_str = " -> ".join(
            f"Player {(i + self.state.game_state['starting_player']) % 2}: [{action}]" for i, action in enumerate(self.history)
        )
        return history_str
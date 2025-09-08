import re, random
from typing import Tuple, Dict, Any, Optional

import textarena as ta
from textarena.envs.KuhnPoker.renderer import create_board_str

from utils.prompt import prompt_template
DEBUGGING = False

class KuhnPokerEnv(ta.Env):
    def __init__(self, max_rounds: int = 1, changing_starting_player: bool = True, 
                 player_0_card: Optional[int] = None, player_1_card: Optional[int] = None):
        super().__init__()
        self.ante = 1
        self.max_rounds = max_rounds
        self.deck = [0, 1, 2]  # 0=J, 1=Q, 2=K
        self.legal_action_tree = {"check": {"check": "showdown", "bet": {"fold": "loser", "call": "showdown2"}}, "bet": {"fold": "loser", "call": "showdown2"}}
        self.player_0_wins = 0
        self.player_1_wins = 0
        self.changing_starting_player = changing_starting_player

        self.new_prompt = prompt_template
        self.history = []
        assert player_0_card in [None, 0, 1, 2] and player_1_card in [None, 0, 1, 2], "player_x_card must be None, 0 (J), 1 (Q), or 2 (K)."
        if player_0_card is not None and player_1_card is not None:
            self.default_deck = [player_0_card, player_1_card, 3-player_0_card-player_1_card]
            self.deck = self.default_deck.copy()
        else:
            self.default_deck = None

    def get_board_str(self): return create_board_str(self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed, error_allowance=0)
        game_state = {"pot": None, "player_chips": {0: 0, 1: 0}, "current_round": 0, "starting_player": 1}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._init_round() # Initialize the first round

    def _init_round(self):
        self.history = []

        self.state.game_state["current_round"] += 1
        if self.state.game_state["current_round"] > self.max_rounds: # check if game is complete
            # determine winner 
            # print(f"The final scores are: Player 0: '{self.state.game_state['player_chips'][0]}'; Player 1: '{self.state.game_state['player_chips'][1]}'", flush=True)
            if self.state.game_state["player_chips"][0] > self.state.game_state["player_chips"][1]: self.state.set_winner(player_id=0, reason=f"Player 0 won by having more chips at the end of all {self.max_rounds} rounds.")
            elif self.state.game_state["player_chips"][0] < self.state.game_state["player_chips"][1]: self.state.set_winner(player_id=1, reason=f"Player 1 won by having more chips at the end of all {self.max_rounds} rounds.")
            else: self.state.set_draw(reason=f"At the end of {self.max_rounds} rounds, both players had the same number of chips.")

        if self.state.done: return

        if DEBUGGING: print(f"### Starting round {self.state.game_state['current_round']} out of {self.max_rounds} rounds.", flush=True)
        if self.default_deck is None:
            random.shuffle(self.deck) # shuffle the deck 
        self.state.game_state["player_cards"] = {0: self.deck[0], 1: self.deck[1]} # assign player cards
        # reset pot
        self.state.game_state["pot"] = self.ante * 2
        self.state.game_state["player_chips"][0] -= self.ante
        self.state.game_state["player_chips"][1] -= self.ante
        # increment round counter
        self.state.game_state["current_legal_action_tree"] = self.legal_action_tree.copy()

        # set starting player
        if not self.changing_starting_player:
            starting_player = 0
        else:
            starting_player = 1 - self.state.game_state["starting_player"]
        self.state.game_state["starting_player"] = starting_player 
        self.state.manually_set_current_player_id(new_player_id=starting_player)

        ## disenable state.add_observation
        # for player_id in range(2):
        #     # message = f"### Starting round {self.state.game_state['current_round']} out of {self.max_rounds} rounds. Your card is: '{self._rank_to_str(self.state.game_state['player_cards'][player_id])}'"
        #     message = f"Your card is: '{self._rank_to_str(self.state.game_state['player_cards'][player_id])}'"
        #     self.state.add_observation(message=message, to_id=player_id, observation_type=ta.ObservationType.GAME_MESSAGE)
        #     if player_id == starting_player:
        #         message = f"Your available actions are: " + ', '.join(f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys())
        #         self.state.add_observation(to_id=player_id, message=message, observation_type=ta.ObservationType.GAME_BOARD)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        # deserted, we only use new_prompt
        return (
            f"You are Player {player_id} in Kuhn Poker.\n"
            f"Game Rules:\n"
            f"- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)\n"
            f"- Each player antes {self.ante} chip and receives 1 card each round "
            f"(note that the cards are dealt without replacement, so you cannot have the same card as your opponent).\n"
            f"- The player with the highest card wins the pot\n\n"
            f"Action Rules:\n"
            f"- '[check]': Pass without betting (only if no bet is on the table)\n"
            f"- '[bet]': Add 1 chip to the pot (only if no bet is on the table)\n"
            f"- '[call]': Match an opponent's bet by adding 1 chip to the pot\n"
            f"- '[fold]': Surrender your hand and let your opponent win the pot\n"
            f"- Note: You must respond with one of the actions in square brackets, '[ACTION]'.\n"
        )
        return (
            f"You are Player {player_id} in a {self.max_rounds} round game of Kuhn Poker.\n"
            f"Game Rules:\n"
            f"- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)\n"
            f"- Each player antes {self.ante} chip and receives 1 card each round "
            f"(note that the cards are dealt without replacement, so you cannot have the same card as your opponent).\n"
            f"- Game continues for {self.max_rounds} rounds\n"
            f"- The player with the most chips after all rounds wins\n\n"
            f"Action Rules:\n"
            f"- '[check]': Pass without betting (only if no bet is on the table)\n"
            f"- '[bet]': Add 1 chip to the pot (only if no bet is on the table)\n"
            f"- '[call]': Match an opponent's bet by adding 1 chip to the pot\n"
            f"- '[fold]': Surrender your hand and let your opponent win the pot\n"
        )

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        rotate_player = True
        # self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        match = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE).search(action.strip()) # Regular expression to capture valid actions: e.g. [Check], [Bet], [Fold], [Call]
        if not match: # Invalid action
            if DEBUGGING: print(f"Invalid action: {action}. Valid actions are: [Check], [Bet], [Fold], [Call].", flush=True)
            # raise ValueError(f"Invalid action: {action}. Valid actions are: [Check], [Bet], [Fold], [Call].")
            # self.state.set_invalid_move(reason="Action must be [Check], [Bet], [Call], or [Fold].")
            # return self.state.step()
            self._set_round_winner(player_id=1-self.state.current_player_id, reason=f"Player {self.state.current_player_id} has invalid action '{action}'."); rotate_player=False
            return self.state.step(rotate_player=rotate_player)

        move = match.group(1).lower()  # 'check', 'bet', 'fold', 'call'
        if move not in self.state.game_state["current_legal_action_tree"].keys():
            if DEBUGGING: print(f"Invalid action: {action}. Valid actions are: {', '.join(self.state.game_state['current_legal_action_tree'].keys())}.", flush=True)
            # raise ValueError(f"Invalid action: {action}. Valid actions are: {', '.join(self.state.game_state['current_legal_action_tree'].keys())}.")
            # legal_actions = ', '.join([f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys()])
            # self.state.set_invalid_move(reason=f"Action must be {legal_actions}.")
            # return self.state.step()
            self._set_round_winner(player_id=1-self.state.current_player_id, reason=f"Player {self.state.current_player_id} has invalid action '{action}'."); rotate_player=False
            return self.state.step(rotate_player=rotate_player)

        self.history.append(move)  # Store the action in history
        step_info = {
            "player_id": self.state.current_player_id, 
            "player_card": self._rank_to_str(self.state.game_state["player_cards"][self.state.current_player_id]), 
            "history": self.history.copy() # include the current action(history[-1])
        }
        # execute move
        # self.state.add_observation(message=f"Player {self.state.current_player_id}, submitted move: '[{move}]'.", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
        self.state.game_state["current_legal_action_tree"] = self.state.game_state["current_legal_action_tree"][move]
        # check if round loser / showdown
        if self.state.game_state["current_legal_action_tree"] == "loser":
            self._set_round_winner(player_id=1-self.state.current_player_id, reason=f"Player {self.state.current_player_id} has folded."); rotate_player=False
        elif self.state.game_state["current_legal_action_tree"] == "showdown":
            self._handle_showdown(); rotate_player=False
        elif self.state.game_state["current_legal_action_tree"] == "showdown2":
            self.state.game_state["pot"] += self.ante * 2 # both players bet/call
            self.state.game_state["player_chips"][0] -= self.ante
            self.state.game_state["player_chips"][1] -= self.ante
            self._handle_showdown(); rotate_player=False
        else: # show valid next actions
            pass
            # legal_actions = ', '.join([f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys()])
            # self.state.add_observation(to_id=1-self.state.current_player_id, message=f"Your available actions are: {legal_actions}", observation_type=ta.ObservationType.GAME_BOARD)
        # return self.state.step(rotate_player=rotate_player)
        return self.state.step(rotate_player=rotate_player)[0], step_info

    def _set_round_winner(self, player_id: int, reason: str):
        self.state.game_state["player_chips"][player_id] += self.state.game_state["pot"]
        if player_id == 0:
            self.player_0_wins += 1
        else:
            self.player_1_wins += 1
        reason += f" Current scores: Player 0: '{self.state.game_state['player_chips'][0]}'; Player 1: '{self.state.game_state['player_chips'][1]}'"
        # self.state.add_observation(message=reason, observation_type=ta.ObservationType.GAME_MESSAGE) # initialize the next cound

        # # clear observations
        # self.state.observations = {pid: [] for pid in range(self.state.num_players)}
        # for pid in range(self.state.num_players):
        #     self.state.add_observation(to_id=pid, message=self._prompt(player_id=pid, game_state=self.state.game_state), observation_type=ta.ObservationType.PROMPT)
        if DEBUGGING: print(f"### Round {self.state.game_state['current_round']} ended. {reason}", flush=True)

        self._init_round() # start next round

    def _rank_to_str(self, rank: int) -> str:
        """Convert the numeric rank to a string 'J', 'Q', or 'K'."""
        return {0: 'J', 1: 'Q', 2: 'K'}.get(rank, '?')

    def _handle_showdown(self):
        card_p0, card_p1 = self.state.game_state["player_cards"][0], self.state.game_state["player_cards"][1]
        winner = 0 if card_p0 > card_p1 else 1 # Determine and announce the winner
        winner_card, loser_card = (card_p0, card_p1) if winner == 0 else (card_p1, card_p0)
        reason = (
            f"Showdown: Player {winner}'s {self._rank_to_str(winner_card)} beats "
            f"Player {1 - winner}'s {self._rank_to_str(loser_card)}. "
            f"Player {winner} wins pot of {self.state.game_state['pot']} chips."
        )
        self._set_round_winner(player_id=winner, reason=reason)


    def get_observation(self):
        """
        Returns the current player's observation string, including their card, action history, and available actions.
        Example format:
        You are Player {player_id} ({first} to act this round).
        Your card: '{card}'
        History: {history}
        Available actions: {available_actions}
        """
        return self.state.current_player_id, self.new_prompt.format(
            player_id=self.state.current_player_id,
            first="first" if self.state.game_state["starting_player"] == self.state.current_player_id else "second",
            card=self._rank_to_str(self.state.game_state["player_cards"][self.state.current_player_id]),
            history=self.get_history_str(),
            available_actions=', '.join([f"[{k}]" for k in self.state.game_state["current_legal_action_tree"].keys()])
        )
    
    def get_history_str(self):
        if not self.history:
            return ""
        # Player 0: [action1] -> Player 1: [action2] -> Player 0: [action3] ...
        history_str = " -> ".join(
            f"Player {(i + self.state.game_state['starting_player']) % 2}: [{action}]" for i, action in enumerate(self.history)
        )
        return history_str

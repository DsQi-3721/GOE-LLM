from textarena.core import Agent

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random

class RandomAgent(Agent):
    def __call__(self, observation: str) -> str:
        """
        Call the agent with an observation and return a random action.
        :param observation: The observation string from the environment.
        :return: A random action string.
        """
        logger.debug(f"RandomAgent Observation: {observation}")
        actions = observation.split("Your available actions are: ")[-1].strip().split(", ")
        action = random.choice(actions)
        logger.debug(f"RandomAgent Action: {action}")
        return "[check]"

    def __str__(self):
        return "RandomAgent"
    

class GtoAgent(Agent):
    def __init__(self, alpha: float = 1/3):
        super().__init__()
        self.alpha = alpha

        self.first_player_gto_1 = {
            "K": {'bet': 3 * alpha, 'check': 1 - 3 * alpha},
            "Q": {'bet': 0.0, 'check': 1.0},
            "J": {'bet': alpha, 'check': 1 - alpha}
        }
        self.first_player_gto_2 = {
            "K": {'call': 1.0, 'fold': 0.0},
            "Q": {'call': alpha + 1/3, 'fold': 2/3 - alpha},
            "J": {'call': 0.0, 'fold': 1.0}
        }

        self.second_player_gto = {
            "K": {'bet': {'call': 1.0, 'fold': 0.0}, 'check': {'bet': 1.0, 'check': 0.0}},
            "Q": {'bet': {'call': 1/3, 'fold': 1 - 1/3}, 'check': {'bet': 0.0, 'check': 1.0}},
            "J": {'bet': {'call': 0.0, 'fold': 1.0}, 'check': {'bet': 1/3, 'check': 1 - 1/3}}
        }

    def __call__(self, observation: str) -> str:
        """
        Call the agent with an observation and return a GTO action.
        :param observation: The observation string from the environment.
        :return: A GTO action string.
        """
        logger.debug(f"GtoAgent Observation: {observation}")
        my_id = observation.split("You are Player ")[-1].split(" in Kuhn Poker")[0]
        info = observation.split("Your card is: ")[-1]
        my_card = info[1]

        action_seq = self.extract_actions(info)

        rand_num = random.random()
        if len(action_seq) == 0:
            print(f"[DEBUGGING] im the first player, my card is {my_card}, rand num is {rand_num}", flush=True)
            if rand_num < self.first_player_gto_1[my_card]['bet']: 
                action = "[bet]"
            else:
                action = "[check]"
        elif len(action_seq) == 1:
            print(f"[DEBUGGING] im the second player, my card is {my_card}, rand num is {rand_num}, action seq is {action_seq}", flush=True)
            if action_seq[0] == 'bet':
                if rand_num < self.second_player_gto[my_card]['bet']['call']:
                    action = "[call]"
                else:
                    action = "[fold]"
            elif action_seq[0] == 'check':
                if rand_num < self.second_player_gto[my_card]['check']['bet']:
                    action = "[bet]"
                else:
                    action = "[check]"
            else:
                raise ValueError(f"Unexpected action sequence: {action_seq[0]}")
        elif len(action_seq) == 2:
            print(f"[DEBUGGING] im the first player, my card is {my_card}, rand num is {rand_num}, action seq is {action_seq}", flush=True)
            if rand_num < self.first_player_gto_2[my_card]['call']:
                action = "[call]"
            else:
                action = "[fold]"
            
        logger.debug(f"GtoAgent Action: {action}")
        return action

    def __str__(self):
        return "GtoAgent"
    
    def extract_actions(self, info: str) -> list:
        actions = []
        lines = info.split('\n')
        for line in lines:
            if 'submitted move' in line:
                action = line.split('submitted move: ')[-1][2:-3] # Extract action from '[action]'
                actions.append(action)
        return actions

'''
[GAME] You are Player 0 in Kuhn Poker.
Game Rules:
- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)
- Each player antes 1 chip and receives 1 card each round (note that the cards are dealt without replacement, so you cannot have the same card as your opponent).
- The player with the highest card wins the pot

Action Rules:
- '[check]': Pass without betting (only if no bet is on the table)
- '[bet]': Add 1 chip to the pot (only if no bet is on the table)
- '[call]': Match an opponent's bet by adding 1 chip to the pot
- '[fold]': Surrender your hand and let your opponent win the pot

[GAME] Your card is: 'K'
[GAME] Player 1, submitted move: '[bet]'.
[GAME] Your available actions are: '[fold]', '[call]'
'''
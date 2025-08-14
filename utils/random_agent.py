from textarena.core import Agent

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random

def clean_obs(observation: str) -> str:
    """
    Clean the observation string by removing unnecessary parts.
    :param observation: The observation string from the environment.
    :return: A cleaned observation string.
    """
    # Remove the game rules and action rules sections
    return observation.split('Surrender your hand and let your opponent win the pot')[-1].strip()

class RandomAgent(Agent):
    def __call__(self, observation: str) -> str:
        """
        Call the agent with an observation and return a random action.
        :param observation: The observation string from the environment.
        :return: A random action string.
        """
        logger.debug("%s Observation: %r", str(self), clean_obs(observation))
        actions = observation.split("Available actions: ")[-1].split('\n')[0].strip().split(", ")
        action = random.choice(actions)
        logger.debug("%s Action: %r", str(self), action.strip("'"))
        return action

    def __str__(self):
        return "RandomAgent"
    

class GtoAgent(Agent):
    def __init__(self, alpha: float = 0):
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
        logger.debug("%s Observation: %r", str(self), clean_obs(observation))
        info = observation.split("Your card: ", maxsplit=1)[-1]
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
            
        logger.debug("%s Action: %r", str(self), action)
        return action

    def __str__(self):
        return "GtoAgent"
    
    def extract_actions(self, info: str) -> list:
        actions = []
        line = info.split('History: ', maxsplit=1)[-1].split('\n')[0].strip()
        if not line:
            return actions
        lines = line.split(' -> ')
        for line in lines:
            if line.startswith("Player 0: "):
                action = line.split("Player 0: ")[-1].strip()
            elif line.startswith("Player 1: "):
                action = line.split("Player 1: ")[-1].strip()
            else:
                raise ValueError(f"Unexpected line format: {line}")
            if action.startswith('[') and action.endswith(']'):
                actions.append(action[1:-1])
        return actions

'''
You are an expert Kuhn Poker player.

[Game Rules]
- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest).
- Each player antes 1 chip and receives 1 card each round (note that the cards are dealt without replacement, so you cannot have the same card as your opponent).
- The player with the highest card wins the pot.

[Action Rules]
- [check]: Pass without betting (only if no bet is on the table)
- [bet]: Add 1 chip to the pot (only if no bet is on the table)
- [call]: Match an opponent's bet by adding 1 chip to the pot
- [fold]: Surrender your hand and let your opponent win the pot

[State]
You are Player 0 (first to act this round).
Your card: 'J'
History: Player 0: [check] -> Player 1: [bet]
Available actions: [fold], [call]

[Output Format]
``` plaintext
<think> Your thoughts and reasoning </think>
\\boxed{{[ACTION]}}
```

[Important Notes]
1. You must always include the <think> field to outline your reasoning.
2. Your final action [ACTION] must be one of the available actions, in \\boxed{{[ACTION]}} format.
'''
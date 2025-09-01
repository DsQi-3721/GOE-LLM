from textarena.core import Agent

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    def __init__(self, alpha: float = 0, bluffing_counter = False):
        super().__init__()
        self.alpha = alpha
        self.bluffing_counter = bluffing_counter
        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"

        # GTO strategy tables [0, 1/3]
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
        self.agent_name = f"GtoAgent({self.alpha})"

        # Bluffing strategy (as first Player)
        if alpha > 1/3:
            # min(p, 1), max(p, 0)
            self.first_player_gto_1 = {k: {k2: max(0, min(1, v2)) for k2, v2 in v.items()} for k, v in self.first_player_gto_1.items()}
            self.first_player_gto_2 = {k: {k2: max(0, min(1, v2)) for k2, v2 in v.items()} for k, v in self.first_player_gto_2.items()}
            self.agent_name = f"Bluffing({self.alpha})"

        # Counter bluffing strategy (as second Player)
        if bluffing_counter:
            self.alpha = 1/3
            self.first_player_gto_1 = {
                "K": {'bet': 1.0, 'check': 0.0},
                "Q": {'bet': 0.0, 'check': 1.0},
                "J": {'bet': 1/3, 'check': 2/3},
            }
            self.first_player_gto_2 = {
                "K": {'call': 1.0, 'fold': 0.0},
                "Q": {'call': 2/3, 'fold': 1/3},
                "J": {'call': 0.0, 'fold': 1.0}
            }
            # always call/bet with K, Q; always fold/check with J
            self.second_player_gto = {
                "K": {'bet': {'call': 1.0, 'fold': 0.0}, 'check': {'bet': 1.0, 'check': 0.0}},
                "Q": {'bet': {'call': 1.0, 'fold': 0.0}, 'check': {'bet': 1.0, 'check': 0.0}},
                "J": {'bet': {'call': 0.0, 'fold': 1.0}, 'check': {'bet': 0.0, 'check': 1.0}}
            }
            self.agent_name = f"Bluffing_counter({self.alpha})"

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
            # print(f"[DEBUGGING] im the first player, my card is {my_card}, rand num is {rand_num}", flush=True)
            if rand_num < self.first_player_gto_1[my_card]['bet']: 
                action = "[bet]"
            else:
                action = "[check]"
        elif len(action_seq) == 1:
            # print(f"[DEBUGGING] im the second player, my card is {my_card}, rand num is {rand_num}, action seq is {action_seq}", flush=True)
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
            # print(f"[DEBUGGING] im the first player, my card is {my_card}, rand num is {rand_num}, action seq is {action_seq}", flush=True)
            if rand_num < self.first_player_gto_2[my_card]['call']:
                action = "[call]"
            else:
                action = "[fold]"
            
        logger.debug("%s Action: %r", str(self), action)
        return action

    def call_parallel(self, observation: str, n: int) -> list:
        """
        Call the agent with a list of observations and return a list of GTO actions.
        :param observations: A list of observation strings from the environment.
        :return: A list of GTO action strings.
        """
        actions = []
        for i in range(n):
            action = self.__call__(observation)
            actions.append(action)
        return actions

    def __str__(self):
        return self.agent_name
    
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
            assert action.startswith('[') and action.endswith(']'), f"Unexpected action format: {action}"
            actions.append(action[1:-1])
        return actions

def describe_opponent(agent_name: str) -> str:
    """
    Translate agent name string into a natural language description
    for LLM prompt conditioning.
    """
    agent_name = agent_name.strip()

    if agent_name == "RandomAgent":
        return ("The opponent plays completely randomly, without any consistent strategy. "
                "Their actions are unpredictable and not based on card strength.")
    
    if agent_name.startswith("Bluffing_counter"):
        return ("The opponent expects bluffs and therefore calls more often with medium-strength hands (Q). "
                "They rarely fold against bets if they hold K or Q, but fold J consistently. "
                "They are sticky and difficult to bluff.")
    
    if agent_name.startswith("Bluffing"):
        # extract alpha if present
        try:
            alpha = float(agent_name.split("(")[-1].rstrip(")"))
        except Exception:
            alpha = None
        if alpha is not None and alpha >= 0.9:
            return ("The opponent plays an extremely aggressive pure bluffing style. "
                    "They often bet and raise regardless of card strength, even with the weakest hands. "
                    "They are highly exploitable by calling them down with strong hands.")
        elif alpha is not None and alpha >= 0.6:
            return ("The opponent over-bluffs frequently with weak card. "
                    "They are aggressive and bet/call too often, making them exploitable by calling wider.")
        else:
            return ("The opponent plays a somewhat bluff-heavy strategy, "
                    "mixing in more weak bluffs than an equilibrium strategy would.")
    
    if agent_name.startswith("GtoAgent"):
        # parse alpha value
        try:
            alpha = float(agent_name.split("(")[-1].rstrip(")"))
        except Exception:
            alpha = None
        
        # if alpha is None:
        return ("The opponent attempts to play according to equilibrium strategies, "
                "balancing value bets and bluffs.")
        # elif alpha == 0.0:
        #     return ("The opponent plays very conservatively. "
        #             "They almost never bluff and usually fold weak hands. ")
        #             # "They only bet with strong cards, making them exploitable by bluffing more often.")
        # elif abs(alpha - 1/3) < 1e-6:
        #     return ("The opponent plays close to Nash equilibrium (GTO). "
        #             "They balance bluffs with J and value bets with K, and defend with Q appropriately. "
        #             "They are difficult to exploit.")
        # elif alpha < 1/3:
        #     return ("The opponent plays a cautious GTO-like strategy with fewer bluffs. "
        #             "They tend to fold weak hands too often, which makes them exploitable by bluffing.")
        # elif alpha > 1/3:
        #     return ("The opponent plays a bluff-heavy GTO-like strategy, "
        #             "bluffing more than equilibrium would suggest. "
        #             "They can be exploited by calling more frequently.")
    
    if agent_name.startswith("DynamicAgent"):
        # 提取当前策略名称
        current_strategy = agent_name.split("Current: ")[-1].rstrip(")")
        return (f"The opponent is a dynamic agent that switches strategies periodically. "
                f"The current strategy is: {current_strategy}. "
                f"This agent adapts its behavior dynamically, making it harder to predict.")

    # fallback
    raise ValueError(f"Unknown agent name: {agent_name}")
    return "The opponent’s strategy is unknown or unusual."

def rebuild_prompts(parquet_path: str, output_path: str):
    from utils.prompt import prompt_template_opponent
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")

    def rebuild_prompt(row):
        # 取出对手描述
        opponent_desc = describe_opponent(row["extra_info"]["opponent_name"])

        # 格式化 new_prompt
        new_prompt: str = row["prompt"][1]['content']
        # print(f"Original prompt:\n{new_prompt}\n")
        new_prompt = new_prompt.replace("[Output Format]", f"""[Opponent Model]
The opponent is estimated to follow this strategy: {opponent_desc}
You may reason about the opponent's ranges, betting patterns, and card strengths.

[Output Format]""")
        # print(f"Rebuilt prompt:\n{new_prompt}\n")

        # 替换 user content
        prompt = row["prompt"].copy()
        prompt[1] = {
            "role": "user",
            "content": new_prompt,
        }
        return prompt

    df["prompt"] = df.apply(rebuild_prompt, axis=1)
    df.to_parquet(output_path, index=False)
    print(f"Rebuilt prompts and saved to {output_path}")

if __name__ == "__main__":
    # 训练数据
    agent = RandomAgent()       # 随机策略
    print(describe_opponent(str(agent)))
    agent = GtoAgent(alpha=0.0) # gto策略 (alpha=0)
    print(describe_opponent(str(agent)))
    agent = GtoAgent(alpha=1/3) # gto策略 (alpha=1/3)
    print(describe_opponent(str(agent)))
    agent = GtoAgent(alpha=0.5) # 诈唬策略 (alpha=1/2)
    print(describe_opponent(str(agent)))

    # 测试数据（除了已有的训练数据，还有OOD数据）
    agent = GtoAgent(alpha=1/6) # gto策略 (alpha=1/6)
    print(describe_opponent(str(agent)))
    agent = GtoAgent(alpha=1/5) # gto策略 (alpha=1/5)
    print(describe_opponent(str(agent)))
    agent = GtoAgent(alpha=2/3) # 诈唬策略 (alpha=2/3)
    print(describe_opponent(str(agent)))
    agent = GtoAgent(alpha=1.0) # 纯诈唬策略
    print(describe_opponent(str(agent)))

    # # read parquet file
    # rebuild_prompts("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/mixed_train_64k.parquet", 
    #                 "/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/mixedoppo_train_64k.parquet")
    # rebuild_prompts("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/mixed_val_64k.parquet", 
    #                 "/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/mixedoppo_val_64k.parquet")
    
    # import pandas as pd
    # # check the rebuilt prompts
    # df = pd.read_parquet("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/mixedoppo_train_64k.parquet")
    # print(df.iloc[0]["prompt"][1]['content'])
    # df = pd.read_parquet("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/mixedoppo_val_64k.parquet")
    # print(df.iloc[0]["prompt"][1]['content'])
    
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
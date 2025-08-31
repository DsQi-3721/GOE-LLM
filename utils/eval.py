import textarena as ta
from utils.random_agent import RandomAgent, GtoAgent
import sys
import re

class DecisionHistory():
    def __init__(self):
        self.tree = {
            "as_player_0": {
                "K": {},
                "Q": {},
                "J": {},
            },
            "as_player_1": {
                "K": {},
                "Q": {},
                "J": {},
            }
        }

def run_eval(eval_agent, opponent_agent, max_rounds=1000):
    dh = DecisionHistory()
    wrong_action = {0: 0, 1: 0}
    # eval_agent as first player, opponent as second player
    agents = {0: eval_agent, 1: opponent_agent}
    env0 = ta.make(env_id="KuhnPoker-v0", max_rounds=max_rounds, changing_starting_player=False)
    env0.reset(num_players=len(agents))
    done = False 
    while not done:
        player_id, observation = env0.get_observation()
        action = agents[player_id](observation)
        if player_id == 0:
            match = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE).search(action.strip())
            if not match: wrong_action[player_id] += 1
            else:
                action = f'[{match.group(1).lower()}]'
                if action in ["[check]", "[bet]", "[fold]", "[call]"]:
                    card = env0._rank_to_str(env0.state.game_state["player_cards"][player_id])
                    his = '->'.join(env0.history)
                    if his not in dh.tree["as_player_0"][card]:
                        dh.tree["as_player_0"][card][his] = {}
                    dh.tree["as_player_0"][card][his][action.lower()] = dh.tree["as_player_0"][card][his].get(action.lower(), 0) + 1
                else:
                    wrong_action[player_id] += 1
        done, step_info = env0.step(action=action)
    
    # eval_agent as second player, opponent as first player
    agents = {1: eval_agent, 0: opponent_agent}
    env1 = ta.make(env_id="KuhnPoker-v0", max_rounds=max_rounds, changing_starting_player=False)
    env1.reset(num_players=len(agents))
    done = False 
    while not done:
        player_id, observation = env1.get_observation()
        action = agents[player_id](observation)
        if player_id == 1:
            match = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE).search(action.strip())
            if not match: wrong_action[player_id] += 1
            else:
                action = f'[{match.group(1).lower()}]'
                if action in ["[check]", "[bet]", "[fold]", "[call]"]:
                    card = env1._rank_to_str(env1.state.game_state["player_cards"][player_id])
                    his = '->'.join(env1.history)
                    if his not in dh.tree["as_player_1"][card]:
                        dh.tree["as_player_1"][card][his] = {}
                    dh.tree["as_player_1"][card][his][action.lower()] = dh.tree["as_player_1"][card][his].get(action.lower(), 0) + 1
                else:
                    wrong_action[player_id] += 1
        done, step_info = env1.step(action=action)

    eval_agent_metrics = {
        "agent": str(eval_agent),
        "opponent": str(opponent_agent),
        "max_rounds": max_rounds,
        "as_player_0": {
            "win_num": env0.player_0_wins,
            "win_chips": (env0.state.game_state["player_chips"][0] + 1),
            "avg_win_chips": (env0.state.game_state["player_chips"][0] + 1) / env0.max_rounds,
        },
        "as_player_1": {
            "win_num": env1.max_rounds - env1.player_0_wins,
            "win_chips": (env1.state.game_state["player_chips"][1] + 1),
            "avg_win_chips": (env1.state.game_state["player_chips"][1] + 1) / env1.max_rounds,
        },
        "decision_history": dh.tree,
        "wrong_action": wrong_action,
    }
    env0.close(); env1.close()
    return eval_agent_metrics

if __name__ == "__main__":
    
    all_opponents = {
        "random": [RandomAgent()],
        "gto": [GtoAgent(0), GtoAgent(1/6), GtoAgent(1/3)],
        "bluffing": [GtoAgent(1/2), GtoAgent(2/3), GtoAgent(5/6), GtoAgent(1)],
    }

    # model path in cmd line: python -m utils.eval /data/models/Qwen2.5-3B-Instruct
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        from utils.llm_agent import VLLMAgent
        eval_agent = VLLMAgent(temperature=0, model_path=model_path)
    if model_path is None:
        eval_agent = RandomAgent()
        eval_agent = GtoAgent(0)

    for opponent_list in all_opponents.values():
        for opponent in opponent_list:
            assert isinstance(opponent, ta.Agent), f"{opponent} is not an instance of ta.Agent"
            print(f"Loaded opponent: {opponent}")
            metrics = run_eval(eval_agent, opponent)
            print(f"Metrics {eval_agent}-{opponent}: {metrics}")

import textarena as ta
from utils.random_agent import RandomAgent, GtoAgent, describe_opponent
from utils.random_agent import BluffAgent, ValueAgent, PassiveAgent, AggressiveAgent
from utils.dynamic_agent import DynamicAgent
import json
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

def obs_opponent_description(agent, opponent_agent, observation):
    if hasattr(agent, "_model_path") and agent._model_path:# and "mixedoppo" in agent._model_path:
        return observation.replace("[Output Format]", f"""[Opponent Model]
The opponent is estimated to follow this strategy: {describe_opponent(str(opponent_agent))}
You may reason about the opponent's ranges, betting patterns, and card strengths. 

[Output Format]""")
    else:
        return observation

def run_eval(eval_agent, opponent_agent, each_deck_rounds=500):
    all_cards_comb = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]  # (player_0_card, player_1_card)
    dh = DecisionHistory()
    wrong_action = {0: 0, 1: 0}
    call_info = {
        "call_parallel_num_0": 0,
        "decision_num_0": 0,
        "call_parallel_num_1": 0,
        "decision_num_1": 0
    }

    def _record(env, player_id, action):
        call_info[f"decision_num_{player_id}"] += 1
        match = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE).search(action.strip())
        if not match: wrong_action[player_id] += 1
        else:
            action = f'[{match.group(1).lower()}]'
            if action in ["[check]", "[bet]", "[fold]", "[call]"]:
                card = env._rank_to_str(env.state.game_state["player_cards"][player_id])
                his = '->'.join(env.history)
                if his not in dh.tree[f"as_player_{player_id}"][card]:
                    dh.tree[f"as_player_{player_id}"][card][his] = {}
                dh.tree[f"as_player_{player_id}"][card][his][action.lower()] = dh.tree[f"as_player_{player_id}"][card][his].get(action.lower(), 0) + 1
            else:
                wrong_action[player_id] += 1

    # eval_agent as first player, opponent as second player
    env0_win_num = 0
    env0_win_chips = 0
    agents = {0: eval_agent, 1: opponent_agent}
    for player_0_card, player_1_card in all_cards_comb:
        env0_list = [
            ta.make(env_id="KuhnPoker-v0", max_rounds=1, changing_starting_player=False, 
                    player_0_card=player_0_card, player_1_card=player_1_card)
            for _ in range(each_deck_rounds)
        ]
        done0_list = [False] * each_deck_rounds
        for env0 in env0_list: env0.reset(num_players=len(agents))
        obs0_dict = {} # obs(str): [indexes in env0_list]
        while not all(done0_list):
            for i, env0 in enumerate(env0_list):
                if not done0_list[i]:
                    player_id, observation = env0.get_observation()
                    if player_id == 1:
                        action = agents[player_id](observation)
                        done0_list[i], step_info = env0.step(action=action)
                    elif player_id == 0:
                        obs0_dict.setdefault(obs_opponent_description(agents[0], agents[1], observation), []).append(i)
                        
            for obs0_str, idxs in obs0_dict.items():
                call_info["call_parallel_num_0"] += 1
                actions = agents[0].call_parallel(obs0_str, n=len(idxs))
                assert len(actions) == len(idxs), f"Expected {len(idxs)} actions, but got {len(actions)}"
                for idx, action in zip(idxs, actions):
                    _record(env0_list[idx], 0, action)
                    done0_list[idx], step_info = env0_list[idx].step(action=action)
            obs0_dict = {}

        env0_win_num += sum([env0.player_0_wins for env0 in env0_list])
        env0_win_chips += sum([env0.state.game_state["player_chips"][0] for env0 in env0_list])
        assert all(env0.state.game_state['player_chips'][0] + env0.state.game_state['player_chips'][1] == 0 for env0 in env0_list), "Total chips should be 0"

        for env0 in env0_list: env0.close()

    env0_avg_win_chips = env0_win_chips / (each_deck_rounds * len(all_cards_comb))
    
    # eval_agent as second player, opponent as first player
    env1_win_num = 0
    env1_win_chips = 0
    agents = {1: eval_agent, 0: opponent_agent}
    for player_0_card, player_1_card in all_cards_comb:
        env1_list = [
            ta.make(env_id="KuhnPoker-v0", max_rounds=1, changing_starting_player=False, 
                    player_0_card=player_0_card, player_1_card=player_1_card)
            for _ in range(each_deck_rounds)
        ]
        done1_list = [False] * each_deck_rounds
        for env1 in env1_list: env1.reset(num_players=len(agents))
        obs1_dict = {} # obs(str): [indexes in env1_list]

        while not all(done1_list):
            for i, env1 in enumerate(env1_list):
                if not done1_list[i]:
                    player_id, observation = env1.get_observation()
                    if player_id == 0:
                        action = agents[player_id](observation)
                        done1_list[i], step_info = env1.step(action=action)
                    elif player_id == 1:
                        obs1_dict.setdefault(obs_opponent_description(agents[1], agents[0], observation), []).append(i)
                        
            for obs1_str, idxs in obs1_dict.items():
                call_info["call_parallel_num_1"] += 1
                actions = agents[1].call_parallel(obs1_str, n=len(idxs))
                assert len(actions) == len(idxs), f"Expected {len(idxs)} actions, but got {len(actions)}"
                for idx, action in zip(idxs, actions):
                    _record(env1_list[idx], 1, action)
                    done1_list[idx], step_info = env1_list[idx].step(action=action)
            obs1_dict = {}

        env1_win_num += sum([env1.player_1_wins for env1 in env1_list])
        env1_win_chips += sum([env1.state.game_state["player_chips"][1] for env1 in env1_list])
        assert all(env1.state.game_state['player_chips'][0] + env1.state.game_state['player_chips'][1] == 0 for env1 in env1_list), "Total chips should be 0"

        for env1 in env1_list: env1.close()

    env1_avg_win_chips = env1_win_chips / (each_deck_rounds * len(all_cards_comb))

    eval_agent_metrics = {
        "agent": str(eval_agent),
        "opponent": str(opponent_agent),
        "max_rounds": each_deck_rounds * len(all_cards_comb),
        "as_player_0": {
            "win_num": env0_win_num,
            "win_chips": env0_win_chips,
            "avg_win_chips": env0_avg_win_chips,
        },
        "as_player_1": {
            "win_num": env1_win_num,
            "win_chips": env1_win_chips,
            "avg_win_chips": env1_avg_win_chips,
        },
        "decision_history": dh.tree,
        "wrong_action": wrong_action,
        "call_info": call_info,
    }
    return eval_agent_metrics

if __name__ == "__main__":
    
    all_opponents = {
        "random": [RandomAgent()],
        "gto": [GtoAgent(0), GtoAgent(1/6), GtoAgent(1/3)],
        "bluffing": [BluffAgent(1/2), BluffAgent(2/3), BluffAgent(5/6), BluffAgent(1)],
        "bluffing_2": [BluffAgent(1/2, 1/2), BluffAgent(2/3, 2/3), BluffAgent(5/6, 5/6), BluffAgent(1, 1)],
        # "bluffing_2": [BluffAgent(1/2, 1/2), BluffAgent(1/2, 5/6),  BluffAgent(1/2, 1),
        #                BluffAgent(5/6, 1/2), BluffAgent(5/6, 5/6), BluffAgent(5/6, 1),
        #                BluffAgent(1, 1/2),   BluffAgent(1, 5/6),   BluffAgent(1, 1)],
        "value_betting": [ValueAgent(0), ValueAgent(1/3), ValueAgent(1/2), ValueAgent(2/3), ValueAgent(1)],
        "passive": [PassiveAgent(0), PassiveAgent(1/3), PassiveAgent(1/2), PassiveAgent(2/3), PassiveAgent(1)],
        "aggressive": [AggressiveAgent(0), AggressiveAgent(1/3), AggressiveAgent(1/2), AggressiveAgent(2/3), AggressiveAgent(1)],
    }

    # model path in cmd line: python -m utils.eval /data/models/Qwen2.5-3B-Instruct
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        from utils.llm_agent import VLLMAgent
        eval_agent = VLLMAgent(model_path=model_path)
        # eval_agent = GtoAgent(0.22)
    else:
        # eval_agent = GtoAgent(0.22)
        assert False, "Please provide the model path in command line: python -m utils.eval /data/models/Qwen2.5-3B-Instruct"

    for opponent_list in all_opponents.values():
        for opponent in opponent_list:
            assert isinstance(opponent, ta.Agent), f"{opponent} is not an instance of ta.Agent"
            print(f"Loaded opponent: {opponent}")
            metrics = run_eval(eval_agent, opponent)
            print(f"Metrics {eval_agent}-{opponent}: {metrics}")

import textarena as ta
from utils.leduc_agents import (
    LeducRandomAgent, LeducGTOAgent, LeducTightAgent, 
    LeducLooseAgent, LeducAggressiveAgent, LeducPassiveAgent,
    describe_leduc_opponent
)
import json
import sys
import re

class LeducDecisionHistory():
    def __init__(self):
        # Leduc Hold'em使用数字表示牌面：0=J, 1=Q, 2=K
        # 支持单张牌和牌对组合
        self.tree = {
            "as_player_0": {},
            "as_player_1": {}
        }

def create_opponent_for_position(opponent_agent, target_player_id):
    """
    根据目标位置创建合适的对手智能体
    
    Args:
        opponent_agent: 原始对手智能体
        target_player_id: 目标玩家位置 (0或1)
    
    Returns:
        适合目标位置的对手智能体
    """
    agent_str = str(opponent_agent)
    
    # 如果是随机智能体，不需要区分先后手
    if "LeducRandomAgent" in agent_str:
        return opponent_agent
    
    # 如果是GTO智能体，需要创建对应player_id的智能体
    if "LeducGTOAgent" in agent_str:
        return LeducGTOAgent(player_id=target_player_id)
    
    # 如果是其他智能体，需要创建对应player_id的智能体
    if "LeducTightAgent" in agent_str:
        # 提取tightness参数
        import re
        tightness_match = re.search(r'\((\d+\.?\d*),player\d+\)', agent_str)
        if tightness_match:
            tightness = float(tightness_match.group(1))
            return LeducTightAgent(tightness=tightness, player_id=target_player_id)
        else:
            return LeducTightAgent(tightness=0.8, player_id=target_player_id)
    
    if "LeducLooseAgent" in agent_str:
        # 提取looseness参数
        import re
        looseness_match = re.search(r'\((\d+\.?\d*),player\d+\)', agent_str)
        if looseness_match:
            looseness = float(looseness_match.group(1))
            return LeducLooseAgent(looseness=looseness, player_id=target_player_id)
        else:
            return LeducLooseAgent(looseness=0.8, player_id=target_player_id)
    
    if "LeducAggressiveAgent" in agent_str:
        # 提取aggression参数
        import re
        aggression_match = re.search(r'\((\d+\.?\d*),player\d+\)', agent_str)
        if aggression_match:
            aggression = float(aggression_match.group(1))
            return LeducAggressiveAgent(aggression=aggression, player_id=target_player_id)
        else:
            return LeducAggressiveAgent(aggression=0.8, player_id=target_player_id)
    
    if "LeducPassiveAgent" in agent_str:
        # 提取passiveness参数
        import re
        passiveness_match = re.search(r'\((\d+\.?\d*),player\d+\)', agent_str)
        if passiveness_match:
            passiveness = float(passiveness_match.group(1))
            return LeducPassiveAgent(passiveness=passiveness, player_id=target_player_id)
        else:
            return LeducPassiveAgent(passiveness=0.8, player_id=target_player_id)
    
    # 默认返回原智能体
    return opponent_agent

def obs_opponent_description(agent, opponent_agent, observation):
    """为LLM agent添加对手描述"""
    if hasattr(agent, "_model_path") and agent._model_path and False: # and False:
        opponent_desc = describe_leduc_opponent(str(opponent_agent))
        return observation.replace("[Output Format]", f"""[Opponent Model]
The opponent is estimated to follow this strategy: {opponent_desc}
You may reason about the opponent's ranges, betting patterns, and card strengths. 

[Output Format]""")
    else:
        return observation

def run_leduc_eval(eval_agent, opponent_agent, total_rounds=300):
    """
    运行Leduc Hold'em评估
    
    Args:
        eval_agent: 被评估的智能体
        opponent_agent: 对手智能体（应该包含player_id信息）
        total_rounds: 总游戏轮数（Leduc环境随机发牌，不需要预设牌面组合）
    """
    """运行Leduc Hold'em评估"""
    
    dh = LeducDecisionHistory()
    wrong_action = {0: 0, 1: 0}
    call_info = {
        "call_parallel_num_0": 0,
        "decision_num_0": 0,
        "call_parallel_num_1": 0,
        "decision_num_1": 0
    }
    
    # 判断对手类型，决定评估方式
    opponent_str = str(opponent_agent)
    is_random_agent = "LeducRandomAgent" in opponent_str
    
    # 获取对手的player_id（如果不是随机智能体）
    opponent_player_id = None
    if not is_random_agent:
        # 从智能体字符串中提取player_id
        import re
        player_id_match = re.search(r'player(\d+)', opponent_str)
        if player_id_match:
            opponent_player_id = int(player_id_match.group(1))
        else:
            # 默认player_id=0
            opponent_player_id = 0

    def _record(env, player_id, action):
        """记录决策历史"""
        call_info[f"decision_num_{player_id}"] += 1
        import re
        match = re.compile(r"\[(Check|Bet|Call|Raise|Fold)\]", re.IGNORECASE).search(action.strip())
        if not match: 
            wrong_action[player_id] += 1
        else:
            action = f'[{match.group(1).lower()}]'
            if action in ["[check]", "[bet]", "[call]", "[raise]", "[fold]"]:
                # 获取手牌信息
                private_card = env.state.game_state["player_cards"][player_id]
                board_card = env.state.game_state.get("board_card", None)
                
                # 构建历史字符串
                his = '->'.join(env.history)
                
                # 构建信息集键 - 使用数字表示牌面
                if board_card is not None:
                    card_key = f"{private_card}{board_card}"
                else:
                    card_key = str(private_card)
                
                if card_key not in dh.tree[f"as_player_{player_id}"]:
                    dh.tree[f"as_player_{player_id}"][card_key] = {}
                if his not in dh.tree[f"as_player_{player_id}"][card_key]:
                    dh.tree[f"as_player_{player_id}"][card_key][his] = {}
                
                dh.tree[f"as_player_{player_id}"][card_key][his][action.lower()] = \
                    dh.tree[f"as_player_{player_id}"][card_key][his].get(action.lower(), 0) + 1
            else:
                wrong_action[player_id] += 1

    # 初始化结果变量
    env0_win_num = 0
    env0_win_chips = 0
    env1_win_num = 0
    env1_win_chips = 0
    
    # 分批处理游戏轮数，每批处理500轮
    batch_size = 60
    num_batches = total_rounds // batch_size
    
    # 根据对手类型决定评估方式
    if is_random_agent:
        # 随机智能体：进行双重评估（先手+后手）
        print(f"随机智能体：开始双重评估，总共{num_batches}批，每批{batch_size}轮")
        
        # 阶段1：eval_agent作为玩家0，对手作为玩家1
        print(f"开始评估作为玩家0，总共{num_batches}批，每批{batch_size}轮")
        agents = {0: eval_agent, 1: opponent_agent}
        
        for batch in range(num_batches):
            print(f"处理批次 {batch+1}/{num_batches}")
            env0_list = [
                ta.make(env_id="LeducHoldem-v0", max_rounds=2, changing_starting_player=False)
                for _ in range(batch_size)
            ]
            done0_list = [False] * batch_size
            for env0 in env0_list: 
                env0.reset(num_players=len(agents))
            
            obs0_dict = {} # obs(str): [indexes in env0_list]
            
            while not all(done0_list):
                for i, env0 in enumerate(env0_list):
                    if not done0_list[i]:
                        player_id, observation = env0.get_observation()
                        if player_id == 1:
                            action = agents[player_id](observation)
                            done0_list[i], step_info = env0.step(action=action)
                        elif player_id == 0:
                            obs0_dict.setdefault(obs_opponent_description(agents[0], opponent_agent, observation), []).append(i)
                            
                for obs0_str, idxs in obs0_dict.items():
                    call_info["call_parallel_num_0"] += 1
                    actions = agents[0].call_parallel(obs0_str, n=len(idxs))
                    assert len(actions) == len(idxs), f"Expected {len(idxs)} actions, but got {len(actions)}"
                    for idx, action in zip(idxs, actions):
                        _record(env0_list[idx], 0, action)
                        done0_list[idx], step_info = env0_list[idx].step(action=action)
                obs0_dict = {}

            env0_win_num += sum([env0.player_0_wins for env0 in env0_list])
            env0_win_chips += sum([env0.state.game_state["player_bank"][0] for env0 in env0_list])

            for env0 in env0_list: 
                env0.close()

        # 阶段2：eval_agent作为玩家1，对手作为玩家0
        print(f"开始评估作为玩家1，总共{num_batches}批，每批{batch_size}轮")
        agents = {1: eval_agent, 0: opponent_agent}
        
        for batch in range(num_batches):
            print(f"处理批次 {batch+1}/{num_batches}")
            env1_list = [
                ta.make(env_id="LeducHoldem-v0", max_rounds=2, changing_starting_player=False)
                for _ in range(batch_size)
            ]
            done1_list = [False] * batch_size
            for env1 in env1_list: 
                env1.reset(num_players=len(agents))
            
            obs1_dict = {} # obs(str): [indexes in env1_list]

            while not all(done1_list):
                for i, env1 in enumerate(env1_list):
                    if not done1_list[i]:
                        player_id, observation = env1.get_observation()
                        if player_id == 0:
                            action = agents[player_id](observation)
                            done1_list[i], step_info = env1.step(action=action)
                        elif player_id == 1:
                            obs1_dict.setdefault(obs_opponent_description(agents[1], opponent_agent, observation), []).append(i)
                            
                for obs1_str, idxs in obs1_dict.items():
                    call_info["call_parallel_num_1"] += 1
                    actions = agents[1].call_parallel(obs1_str, n=len(idxs))
                    assert len(actions) == len(idxs), f"Expected {len(idxs)} actions, but got {len(actions)}"
                    for idx, action in zip(idxs, actions):
                        _record(env1_list[idx], 1, action)
                        done1_list[idx], step_info = env1_list[idx].step(action=action)
                obs1_dict = {}

            env1_win_num += sum([env1.player_1_wins for env1 in env1_list])
            env1_win_chips += sum([env1.state.game_state["player_bank"][1] for env1 in env1_list])

            for env1 in env1_list: 
                env1.close()
                
    else:
        # 非随机智能体：根据player_id进行单次评估
        if opponent_player_id == 0:
            # 对手作为玩家0，eval_agent作为玩家1
            print(f"对手作为玩家0，eval_agent作为玩家1，总共{num_batches}批，每批{batch_size}轮")
            agents = {0: opponent_agent, 1: eval_agent}
            
            for batch in range(num_batches):
                print(f"处理批次 {batch+1}/{num_batches}")
                env1_list = [
                    ta.make(env_id="LeducHoldem-v0", max_rounds=2, changing_starting_player=False)
                    for _ in range(batch_size)
                ]
                done1_list = [False] * batch_size
                for env1 in env1_list: 
                    env1.reset(num_players=len(agents))
                
                obs1_dict = {} # obs(str): [indexes in env1_list]

                while not all(done1_list):
                    for i, env1 in enumerate(env1_list):
                        if not done1_list[i]:
                            player_id, observation = env1.get_observation()
                            if player_id == 0:
                                action = agents[player_id](observation)
                                done1_list[i], step_info = env1.step(action=action)
                            elif player_id == 1:
                                obs1_dict.setdefault(obs_opponent_description(agents[1], opponent_agent, observation), []).append(i)
                                
                    for obs1_str, idxs in obs1_dict.items():
                        call_info["call_parallel_num_1"] += 1
                        actions = agents[1].call_parallel(obs1_str, n=len(idxs))
                        assert len(actions) == len(idxs), f"Expected {len(idxs)} actions, but got {len(actions)}"
                        for idx, action in zip(idxs, actions):
                            _record(env1_list[idx], 1, action)
                            done1_list[idx], step_info = env1_list[idx].step(action=action)
                    obs1_dict = {}

                env1_win_num += sum([env1.player_1_wins for env1 in env1_list])
                env1_win_chips += sum([env1.state.game_state["player_bank"][1] for env1 in env1_list])

                for env1 in env1_list: 
                    env1.close()
                    
        else:
            # 对手作为玩家1，eval_agent作为玩家0
            print(f"对手作为玩家1，eval_agent作为玩家0，总共{num_batches}批，每批{batch_size}轮")
            agents = {0: eval_agent, 1: opponent_agent}
            
            for batch in range(num_batches):
                print(f"处理批次 {batch+1}/{num_batches}")
                env0_list = [
                    ta.make(env_id="LeducHoldem-v0", max_rounds=2, changing_starting_player=False)
                    for _ in range(batch_size)
                ]
                done0_list = [False] * batch_size
                for env0 in env0_list: 
                    env0.reset(num_players=len(agents))
                
                obs0_dict = {} # obs(str): [indexes in env0_list]
                
                while not all(done0_list):
                    for i, env0 in enumerate(env0_list):
                        if not done0_list[i]:
                            player_id, observation = env0.get_observation()
                            if player_id == 1:
                                action = agents[player_id](observation)
                                done0_list[i], step_info = env0.step(action=action)
                            elif player_id == 0:
                                obs0_dict.setdefault(obs_opponent_description(agents[0], opponent_agent, observation), []).append(i)
                                
                    for obs0_str, idxs in obs0_dict.items():
                        call_info["call_parallel_num_0"] += 1
                        actions = agents[0].call_parallel(obs0_str, n=len(idxs))
                        assert len(actions) == len(idxs), f"Expected {len(idxs)} actions, but got {len(actions)}"
                        for idx, action in zip(idxs, actions):
                            _record(env0_list[idx], 0, action)
                            done0_list[idx], step_info = env0_list[idx].step(action=action)
                    obs0_dict = {}

                env0_win_num += sum([env0.player_0_wins for env0 in env0_list])
                env0_win_chips += sum([env0.state.game_state["player_bank"][0] for env0 in env0_list])

                for env0 in env0_list: 
                    env0.close()

    # 计算平均筹码
    env0_avg_win_chips = env0_win_chips / total_rounds if env0_win_chips > 0 else 0
    env1_avg_win_chips = env1_win_chips / total_rounds if env1_win_chips > 0 else 0

    # 根据评估类型构建结果
    if is_random_agent:
        # 随机智能体：返回双重评估结果
        eval_agent_metrics = {
            "agent": str(eval_agent),
            "opponent": str(opponent_agent),
            "evaluation_type": "dual_position",
            "max_rounds": total_rounds,
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
    else:
        # 非随机智能体：根据player_id返回单次评估结果
        if opponent_player_id == 0:
            # 对手作为玩家0，eval_agent作为玩家1
            eval_agent_metrics = {
                "agent": str(eval_agent),
                "opponent": str(opponent_agent),
                "evaluation_type": "single_position",
                "opponent_position": 0,
                "eval_agent_position": 1,
                "max_rounds": total_rounds,
                "as_player_1": {
                    "win_num": env1_win_num,
                    "win_chips": env1_win_chips,
                    "avg_win_chips": env1_avg_win_chips,
                },
                "decision_history": dh.tree,
                "wrong_action": wrong_action,
                "call_info": call_info,
            }
        else:
            # 对手作为玩家1，eval_agent作为玩家0
            eval_agent_metrics = {
                "agent": str(eval_agent),
                "opponent": str(opponent_agent),
                "evaluation_type": "single_position",
                "opponent_position": 1,
                "eval_agent_position": 0,
                "max_rounds": total_rounds,
                "as_player_0": {
                    "win_num": env0_win_num,
                    "win_chips": env0_win_chips,
                    "avg_win_chips": env0_avg_win_chips,
                },
                "decision_history": dh.tree,
                "wrong_action": wrong_action,
                "call_info": call_info,
            }
    
    return eval_agent_metrics

if __name__ == "__main__":
    
    all_opponents = {
        "random": [LeducRandomAgent()],  # 随机智能体：自动进行双重评估
        "gto": [LeducGTOAgent(player_id=0), LeducGTOAgent(player_id=1)],  # 根据player_id进行单次评估
        #"tight": [LeducTightAgent(tightness=0.6, player_id=0), LeducTightAgent(tightness=0.8, player_id=0), LeducTightAgent(tightness=1.0, player_id=0),
        #         LeducTightAgent(tightness=0.6, player_id=1), LeducTightAgent(tightness=0.8, player_id=1), LeducTightAgent(tightness=1.0, player_id=1)],
        #"loose": [LeducLooseAgent(looseness=0.6, player_id=0), LeducLooseAgent(looseness=0.8, player_id=0), LeducLooseAgent(looseness=1.0, player_id=0),
        #         LeducLooseAgent(looseness=0.6, player_id=1), LeducLooseAgent(looseness=0.8, player_id=1), LeducLooseAgent(looseness=1.0, player_id=1)],
        #"aggressive": [LeducAggressiveAgent(aggression=0.6, player_id=0), LeducAggressiveAgent(aggression=0.8, player_id=0), LeducAggressiveAgent(aggression=1.0, player_id=0),
        #              LeducAggressiveAgent(aggression=0.6, player_id=1), LeducAggressiveAgent(aggression=0.8, player_id=1), LeducAggressiveAgent(aggression=1.0, player_id=1)],
        #"passive": [LeducPassiveAgent(passiveness=0.6, player_id=0), LeducPassiveAgent(passiveness=0.8, player_id=0), LeducPassiveAgent(passiveness=1.0, player_id=0),
        #           LeducPassiveAgent(passiveness=0.6, player_id=1), LeducPassiveAgent(passiveness=0.8, player_id=1), LeducPassiveAgent(passiveness=1.0, player_id=1)],
    }

    # model path in cmd line: python -m utils.eval_leduc_parallel /data/models/Qwen2.5-3B-Instruct
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"开始加载模型: {model_path}")
        from utils.llm_agent import VLLMAgent
        eval_agent = VLLMAgent(model_path=model_path)
        print("模型加载完成!")
    else:
        assert False, "Please provide the model path in command line: python -m utils.eval_leduc_parallel /data/models/Qwen2.5-3B-Instruct"

    print("开始评估所有对手...")
    total_opponents = sum(len(opponent_list) for opponent_list in all_opponents.values())
    current_opponent = 0
    
    for category, opponent_list in all_opponents.items():
        print(f"评估类别: {category}")
        for opponent in opponent_list:
            current_opponent += 1
            assert isinstance(opponent, ta.Agent), f"{opponent} is not an instance of ta.Agent"
            print(f"[{current_opponent}/{total_opponents}] 开始评估对手: {opponent}")
            metrics = run_leduc_eval(eval_agent, opponent)
            print(f"[{current_opponent}/{total_opponents}] 完成评估: {eval_agent}-{opponent}")
            print(f"结果: {metrics}")
            print("-" * 50)

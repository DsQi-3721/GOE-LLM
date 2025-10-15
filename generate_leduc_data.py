#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Leduc Hold'em对局训练数据
参考generate_data.py的结构，为Leduc环境生成不同策略和best response对手的对局数据
"""

import sys
import os
import json
import copy
import random
import pandas as pd
from typing import List, Dict, Any

# 添加路径
sys.path.insert(0, '/home/wangxinqi/llm-opponent-modeling')

from utils.leduc_agents import (
    LeducRandomAgent, LeducGTOAgent, LeducTightAgent, 
    LeducLooseAgent, LeducAggressiveAgent, LeducPassiveAgent
)
from textarena.envs.LeducHoldem.env import LeducHoldemEnv
from textarena.agents.basic_agents import STANDARD_GAME_PROMPT

def write_data_to_file(data, filename):
    """
    Write the generated data to a JSON file.
    
    Args:
        data (list): The data to write.
        filename (str): The name of the file to write to.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def generate_data(start_index, agent1, agent2, total_rounds=32000, changing_starting_player=True, collect_agents=[]):
    """
    Generate data for Leduc Hold'em game using different agent strategies.
    The data will be saved in a JSON file.
    
    Args:
        start_index: Starting index for data items
        agent1: Agent for player 0
        agent2: Agent for player 1
        total_rounds: Total number of rounds to play
        changing_starting_player: Whether to change starting player each hand
        collect_agents: List of player IDs to collect data from
    """
    new_data = []
    agents = {
        0: agent1,
        1: agent2,
    }

    env = LeducHoldemEnv(
        starting_bank=total_rounds * 20,  # 增加起始银行余额，减少破产概率
        max_rounds=total_rounds, 
        changing_starting_player=changing_starting_player
    )

    env.reset(num_players=len(agents))
    done = False

    """
    collect data

    data = {
        "data_source": "textarena/leduc_holdem",
        "prompt": [
            {   "role": "system",
                "content": STANDARD_GAME_PROMPT
            },
            {
                "role": "user",
                "content": observation,
            }
        ],
        "ability": "leduc_holdem",
        "reward_model": {"style": "rule", "ground_truth": action},
        "extra_info": {
            ...
        },
    }
    """
    while not done:
        player_id, observation = env.get_observation()
        
        # Extract game state information
        gs = env.state.game_state
        player_card = env._rank_to_str(gs["player_cards"][player_id])
        opponent_card = env._rank_to_str(gs["player_cards"][1 - player_id])
        board_card = env._rank_to_str(gs["board_card"]) if gs["round"] > 0 else None
        
        # Create action history string
        action_history = ('->'.join(env.history) + '->') if env.history else ''
        
        # Copy game state to avoid mutation
        game_state = copy.deepcopy(gs)

        # Get action from agent
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)

        # Collect data if this player is in collect_agents list
        if player_id in collect_agents:
            item = {
                "data_source": "textarena/leduc_holdem",
                "prompt": [
                    {
                        "role": "system",
                        "content": STANDARD_GAME_PROMPT
                    },
                    {
                        "role": "user",
                        "content": observation,
                    }
                ],
                "ability": "leduc_holdem",
                "reward_model": {"style": "rule", "ground_truth": action},
                "extra_info": {
                    "split": "train",
                    "index": len(new_data) + start_index,
                    "player_name": str(agents[player_id]),
                    "opponent_name": str(agents[1 - player_id]),
                    "player_id": player_id,
                    "player_card": player_card,
                    "opponent_card": opponent_card,
                    "board_card": board_card,
                    "round": gs["round"],
                    "action_history": action_history,
                    "game_state": json.dumps(game_state),
                    "step_info": json.dumps(step_info),
                },
            }
            new_data.append(item)

    # Get final game statistics
    final_banks = env.state.game_state["player_bank"]
    win_num = env.player_0_wins
    total_hands = env.state.game_state.get('hands_dealt', 0)
    
    print(f"Game finished. Player 0 bank: {final_banks[0] - total_rounds * 20}, Player 1 bank: {final_banks[1] - total_rounds * 20}")
    print(f"Avg win per hand - Player 0: {(final_banks[0] - total_rounds * 20) / total_hands}, Player 1: {(final_banks[1] - total_rounds * 20) / total_hands}")
    print(f"Total hands: {total_hands}, Player 0 wins: {win_num}, Player 1 wins: {env.player_1_wins}")
    
    # Calculate win rates
    if total_hands > 0:
        win_rate_0 = win_num / total_hands
        win_rate_1 = env.player_1_wins / total_hands
        print(f"Win rates - Player 0: {win_rate_0:.2%}, Player 1: {win_rate_1:.2%}")
    
    # Shuffle data
    random.shuffle(new_data)
    return new_data

def generate_data_mixed():
    """
    Generate mixed data with different agent combinations for Leduc Hold'em.
    This creates diverse training data with various strategy types.
    
    Note: For GTO strategies, we use changing_starting_player=False because:
    - PyCFR strategies are position-specific (player0.strat vs player1.strat)
    - Random position changes would cause strategy-position mismatch
    - For non-GTO strategies, we can use changing_starting_player=True for variety
    """
    data = []
    game_stats = []  # Store game statistics for analysis
    '''
    # 1. Random vs GTO (collect from GTO player)
    print("=" * 60)
    print("1. Generating Random vs GTO data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducRandomAgent(),
        LeducGTOAgent(player_id=1),
        total_rounds=12000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[1]  # Collect from GTO player
    )[:12000]
    data += new_data
    print(f"Generated {len(new_data)} items of Random vs GTO data.")
    print()

    # 1. GTO vs Random (collect from GTO player)
    print("=" * 60)
    print("1. Generating GTO vs Random data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducGTOAgent(player_id=0),
        LeducRandomAgent(),
        total_rounds=12000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[0]  # Collect from GTO player
    )[:12000]
    data += new_data
    print(f"Generated {len(new_data)} items of GTO vs Random data.")
    print()
    '''
    # 2. GTO vs GTO (collect from both players)
    print("=" * 60)
    print("2. Generating GTO vs GTO data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducGTOAgent(player_id=0),
        LeducGTOAgent(player_id=1),
        total_rounds=100000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[0, 1]  # Collect from both players
    )[:32000]
    data += new_data
    print(f"Generated {len(new_data)} items of GTO vs GTO data.")
    print()
    '''
    # 3. Tight vs Loose (collect from both players)
    print("=" * 60)
    print("3. Generating Tight vs Loose data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducTightAgent(player_id=0, tightness=0.8),
        LeducLooseAgent(player_id=1, looseness=0.8),
        total_rounds=10,
        changing_starting_player=False,  # 非GTO策略可以使用位置轮换
        collect_agents=[1]  # Collect from both players
    )[:12000]
    data += new_data
    print(f"Generated {len(new_data)} items of Tight vs Loose data.")
    print()

    # 3. Loose vs Tight  (collect from both players)
    print("=" * 60)
    print("3. Generating Loose vs Tight data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducLooseAgent(player_id=0, looseness=0.8),
        LeducTightAgent(player_id=1, tightness=0.8),
        total_rounds=12000,
        changing_starting_player=False,  # 非GTO策略可以使用位置轮换
        collect_agents=[0]  # Collect from both players
    )[:12000]
    data += new_data
    print(f"Generated {len(new_data)} items of Loose vs Tight data.")
    print()
    '''
    '''
    # 4. Aggressive vs Passive (collect from both players)
    print("=" * 60)
    print("4. Generating Aggressive vs Passive data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducAggressiveAgent(player_id=0, aggression=0.8),
        LeducPassiveAgent(player_id=1, passiveness=0.8),
        total_rounds=12000,
        changing_starting_player=False,  # 非GTO策略可以使用位置轮换
        collect_agents=[0, 1]  # Collect from both players
    )[:12000]
    data += new_data
    print(f"Generated {len(new_data)} items of Aggressive vs Passive data.")
    print()

    # 4. Passive vs Aggressive (collect from both players)
    print("=" * 60)
    print("4. Generating Passive vs Aggressive data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducPassiveAgent(player_id=0, passiveness=0.8),
        LeducAggressiveAgent(player_id=1, aggression=0.8),
        total_rounds=12000,
        changing_starting_player=False,  # 非GTO策略可以使用位置轮换
        collect_agents=[0, 1]  # Collect from both players
    )[:12000]
    data += new_data
    print(f"Generated {len(new_data)} items of Passive vs Aggressive data.")
    print()
    '''
    '''
    # 5. GTO vs Tight (collect from GTO player)
    print("=" * 60)
    print("5. Generating GTO vs Tight data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducGTOAgent(player_id=0),
        LeducTightAgent(player_id=1, tightness=0.7),
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[0]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of GTO vs Tight data.")
    print()
    
    # 5. Tight vs GTO  (collect from GTO player)
    print("=" * 60)
    print("5. Generating Tight vs GTO data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducTightAgent(player_id=0, tightness=0.7),
        LeducGTOAgent(player_id=1),
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[1]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of Tight vs GTO data.")
    print()

    # 6. GTO vs Loose (collect from GTO player)
    print("=" * 60)
    print("6. Generating GTO vs Loose data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducGTOAgent(player_id=0),
        LeducLooseAgent(player_id=1, looseness=0.7),  # 使用更明显的松手特征
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[0]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of GTO vs Loose data.")
    print()

    # 6. Loose vs GTO (collect from GTO player)
    print("=" * 60)
    print("6. Generating Loose vs GTO data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducLooseAgent(player_id=0, looseness=0.7),  # 使用更明显的松手特征
        LeducGTOAgent(player_id=1),
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[1]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of Loose vs GTO data.")
    print()
    '''
    '''
    # 7. GTO vs Aggressive (collect from GTO player)
    print("=" * 60)
    print("7. Generating GTO vs Aggressive data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducGTOAgent(player_id=0),
        LeducAggressiveAgent(player_id=1, aggression=0.7),  # 使用更明显的激进特征
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[0]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of GTO vs Aggressive data.")
    print()
    
    # 7. Aggressive vs GTO (collect from GTO player)
    print("=" * 60)
    print("7. Generating Aggressive vs GTO data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducAggressiveAgent(player_id=0, aggression=0.7),  # 使用更明显的激进特征
        LeducGTOAgent(player_id=1),
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[1]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of Aggressive vs GTO data.")
    print()
    '''
    '''
    # 8. GTO vs Passive (collect from GTO player)
    print("=" * 60)
    print("8. Generating GTO vs Passive data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducGTOAgent(player_id=0),
        LeducPassiveAgent(player_id=1, passiveness=0.9),
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[0]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of GTO vs Passive data.")
    print()

    # 8. Passive vs GTO (collect from GTO player)
    print("=" * 60)
    print("8. Generating Passive vs GTO data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducPassiveAgent(player_id=0, passiveness=0.9),
        LeducGTOAgent(player_id=1),
        total_rounds=10000,
        changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
        collect_agents=[1]  # Collect from GTO player
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of Passive vs GTO data.")
    print()
    '''
    '''
    # 9. Random vs Random (collect from both players for baseline)
    print("=" * 60)
    print("9. Generating Random vs Random data...")
    print("=" * 60)
    new_data = generate_data(
        len(data),
        LeducRandomAgent(),
        LeducRandomAgent(),
        total_rounds=10000,
        changing_starting_player=True,  # Random策略可以使用位置轮换
        collect_agents=[0, 1]  # Collect from both players
    )[:10000]
    data += new_data
    print(f"Generated {len(new_data)} items of Random vs Random data.")
    print()
    '''
    return data

def generate_best_response_data():
    """
    Generate data specifically for best response scenarios.
    This focuses on GTO vs exploitable strategies.
    """
    data = []
    
    print("=" * 80)
    print("GENERATING BEST RESPONSE DATA")
    print("=" * 80)
    
    # GTO vs various exploitable strategies (collect from GTO player)
    exploitable_strategies = [
        ("random", LeducRandomAgent()),
        ("gto", LeducGTOAgent(player_id=1)),
        ("tight0.6", LeducTightAgent(player_id=1, tightness=0.6)),
        ("tight0.8", LeducTightAgent(player_id=1, tightness=0.8)),    
        ("tight1", LeducTightAgent(player_id=1, tightness=1)),        
        ("loose0.6", LeducLooseAgent(player_id=1, looseness=0.6)),    
        ("loose0.8", LeducLooseAgent(player_id=1, looseness=0.8)),    
        ("loose1", LeducLooseAgent(player_id=1, looseness=1)),        
        ("passive0.6", LeducPassiveAgent(player_id=1, passiveness=0.6)), 
        ("passive0.8", LeducPassiveAgent(player_id=1, passiveness=0.8)), 
        ("passive1", LeducPassiveAgent(player_id=1, passiveness=1)),        
        ("aggressive0.6", LeducAggressiveAgent(player_id=1, aggression=0.6)), 
        ("aggressive0.8", LeducAggressiveAgent(player_id=1, aggression=0.8)), 
        ("aggressive1", LeducAggressiveAgent(player_id=1, aggression=1)),        
    ]

    exploitable_strategies_2 = [
        ("random", LeducRandomAgent()),
        ("gto", LeducGTOAgent(player_id=0)),
        ("tight0.6", LeducTightAgent(player_id=0, tightness=0.6)),
        ("tight0.8", LeducTightAgent(player_id=0, tightness=0.8)),    
        ("tight1", LeducTightAgent(player_id=0, tightness=1)),        
        ("loose0.6", LeducLooseAgent(player_id=0, looseness=0.6)),    
        ("loose0.8", LeducLooseAgent(player_id=0, looseness=0.8)),    
        ("loose1", LeducLooseAgent(player_id=0, looseness=1)),        
        ("passive0.6", LeducPassiveAgent(player_id=0, passiveness=0.6)), 
        ("passive0.8", LeducPassiveAgent(player_id=0, passiveness=0.8)), 
        ("passive1", LeducPassiveAgent(player_id=0, passiveness=1)),        
        ("aggressive0.6", LeducAggressiveAgent(player_id=0, aggression=0.6)), 
        ("aggressive0.8", LeducAggressiveAgent(player_id=0, aggression=0.8)), 
        ("aggressive1", LeducAggressiveAgent(player_id=0, aggression=1)),        
    ]
    
    for i, (strategy_name, strategy_agent) in enumerate(exploitable_strategies, 1):
        print("=" * 60)
        print(f"{i}. Generating GTO vs {strategy_name.upper()} (best response) data...")
        print("=" * 60)
        new_data = generate_data(
            len(data),
            LeducAggressiveAgent(player_id=0, aggression=0.8),
            strategy_agent,
            total_rounds=600,
            changing_starting_player=False,  # 固定位置，避免策略与位置不匹配
            collect_agents=[0]  # Collect from GTO player (best responder)
        )[:10000]
        data += new_data
        print(f"Generated {len(new_data)} items of GTO vs {strategy_name} data.")
        print()

    return data

def read_data_from_file(filename):
    """
    Read data from a JSON file.
    Args:
        filename (str): The name of the file to read from.
    Returns:
        list: The data read from the file.
    """
    with open(filename, "r") as f:
        return json.load(f)

def main():
    """
    Main function to generate Leduc Hold'em training data.
    """
    '''
    print("=" * 80)
    print("STARTING LEDUC HOLD'EM DATA GENERATION")
    print("=" * 80)
    
    # Generate mixed data
    print("\n" + "=" * 80)
    print("GENERATING MIXED STRATEGY DATA")
    print("=" * 80)
    data = generate_data_mixed()
    write_data_to_file(data, "leduc_holdem_data_mixed.json")
    print(f"\n✅ Mixed data generation completed!")
    print(f"📊 Total generated data: {len(data)} items")
    '''
    
    # Generate best response data
    br_data = generate_best_response_data()
    write_data_to_file(br_data, "leduc_holdem_best_response_data.json")
    print(f"\n✅ Best response data generation completed!")
    print(f"📊 Total best response data: {len(br_data)} items")
    '''
    # Combine all data
    all_data = data + br_data
    write_data_to_file(all_data, "leduc_holdem_all_data.json")
    print(f"\n✅ All data combined!")
    print(f"📊 Total combined data: {len(all_data)} items")
    
    # Print comprehensive statistics
    print("\n" + "=" * 80)
    print("FINAL DATA GENERATION STATISTICS")
    print("=" * 80)
    
    # Agent combination statistics
    agent_stats = {}
    for item in all_data:
        player_name = item["extra_info"]["player_name"]
        opponent_name = item["extra_info"]["opponent_name"]
        combo = f"{player_name}_vs_{opponent_name}"
        
        if combo not in agent_stats:
            agent_stats[combo] = 0
        agent_stats[combo] += 1
    
    print("\n📈 Agent combination statistics:")
    print("-" * 60)
    for combo, count in sorted(agent_stats.items()):
        percentage = (count / len(all_data)) * 100
        print(f"  {combo:<50} {count:>8} items ({percentage:>5.1f}%)")
    
    print(f"\n🎯 Summary:")
    print(f"  • Total data items: {len(all_data):,}")
    print(f"  • Mixed strategy data: {len(data):,}")
    #print(f"  • Best response data: {len(br_data):,}")
    #print(f"  • Unique agent combinations: {len(agent_stats)}")
    
    print("\n✅ Data generation completed successfully!")
    print("📁 Files saved:")
    print("  • leduc_holdem_data_mixed.json")
    #print("  • leduc_holdem_best_response_data.json") 
    #print("  • leduc_holdem_all_data.json")
    '''
'''
    # Convert to parquet format like generate_data.py
    
    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET FORMAT")
    print("=" * 80)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"📊 DataFrame created with {len(df)} rows")
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    print("🔄 DataFrame shuffled")
    
    # Select subset (like generate_data.py)
    df = df[:80000]  # Limit to 64k items like in generate_data.py
    print(f"📏 Limited to {len(df)} items")
    
    # Split into train/val/test
    train_size = int(64000 * 0.8)
    val_size = int(64000 * 0.2)
    test_size = len(df) - train_size - val_size
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"📊 Split sizes:")
    print(f"  • Train: {len(train_df):,} items")
    print(f"  • Val: {len(val_df):,} items") 
    print(f"  • Test: {len(test_df):,} items")
    
    # Save to parquet files
    train_df.to_parquet("leduc_holdem_train_v2.parquet", index=False)
    val_df.to_parquet("leduc_holdem_val_v2.parquet", index=False)
    test_df.to_parquet("leduc_holdem_test_v2.parquet", index=False)
    
    print("\n✅ Parquet files saved:")
    print("  • leduc_holdem_train.parquet")
    print("  • leduc_holdem_val.parquet")
    print("  • leduc_holdem_test.parquet")
'''
if __name__ == "__main__":
    main()

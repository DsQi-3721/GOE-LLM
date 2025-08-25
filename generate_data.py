from utils.random_agent import GtoAgent, RandomAgent
import textarena as ta
from textarena.agents.basic_agents import STANDARD_GAME_PROMPT
import json
import copy
import random

def write_data_to_file(data, filename):
    """
    Write the generated data to a JSON file.
    
    Args:
        data (list): The data to write.
        filename (str): The name of the file to write to.
    """
    # write data to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def generate_data(start_index, agent1, agent2, total_rounds=32000, changing_starting_player=False, collect_agents=[]):
    """
    Generate data for Kuhn Poker game using GtoAgent.
    The data will be saved in a JSON file.
    """
    new_data = []
    agents = {
        0: agent1,
        1: agent2,
    }

    env = ta.make(
        env_id="KuhnPoker-v0", 
        max_rounds=total_rounds, 
        changing_starting_player=changing_starting_player
    )

    env.reset(num_players=len(agents))
    done = False 

    """
    collect data

    data = {
        "data_source": "textarena/kuhn_poker",
        "prompt": [
            {   "role": "system",
                "content": {STANDARD_GAME_PROMPT}
            },
            {
                "role": "user",
                "content": observation,
            }
        ],
        "ability": "kuhn_poker",
        "reward_model": {"style": "rule", "ground_truth": {action}},
        "extra_info": {
            ...
        },
    }
    """
    while not done:
        player_id, observation = env.get_observation()
        # env.state.game_state is dict, copy it to avoid mutation
        player_card = env._rank_to_str(env.state.game_state["player_cards"][player_id])
        opponent_card = env._rank_to_str(env.state.game_state["player_cards"][1 - player_id])
        action_history = ('->'.join(env.histroy) + '->') if env.histroy else ''
        game_state = copy.deepcopy(env.state.game_state)

        action = agents[player_id](observation)
        done, step_info = env.step(action=action)

        if player_id in collect_agents:
            item = {
                "data_source": "textarena/kuhn_poker",
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
                "ability": "kuhn_poker",
                "reward_model": {"style": "rule", "ground_truth": action},
                "extra_info": {
                    "split": "train",
                    "index": len(new_data) + start_index,
                    "player_name": str(agents[player_id]),
                    "opponent_name": str(agents[1 - player_id]),
                    "player_id": player_id,
                    "player_card": player_card,
                    "opponent_card": opponent_card,
                    "action_history": action_history,
                    "game_state": json.dumps(game_state),
                },
            }
            new_data.append(item)

    win_num = env.player_0_wins
    rewards, game_info = env.close()

    print(f"Game finished. Player 0 chips: {game_state['player_chips'][0]}, Player 1 chips: {game_state['player_chips'][1]}")
    print(f"Total rounds: {total_rounds}, Player 0 wins: {win_num}, Player 1 wins: {total_rounds - win_num}")
    # shuffle data
    random.shuffle(new_data)
    return new_data

def generate_data_mixed():
    data = []
    data += generate_data(
        len(data),
        RandomAgent(),
        GtoAgent(alpha=1/3),
        total_rounds=32000,
        changing_starting_player=True,
        collect_agents=[1]
    )[:32000]
    print(f"Generated {len(data)} items of data.")

    data += generate_data(
        len(data),
        GtoAgent(alpha=1/3),
        GtoAgent(alpha=1/3),
        total_rounds=32000,
        changing_starting_player=True,
        collect_agents=[0,1]
    )[:64000]
    print(f"Generated {len(data)} items of data.")

    data += generate_data(
        len(data),
        GtoAgent(alpha=1/2),
        GtoAgent(bluffing_counter=True),
        total_rounds=32000,
        changing_starting_player=True,
        collect_agents=[1]
    )[:32000]
    print(f"Generated {len(data)} items of data.")

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
    # data = generate_data_mixed()
    # write_data_to_file(data, "kuhn_poker_data_mixed_32k.json")
    # exit()

    data = read_data_from_file("/home/cuisijia/llm_opponent_modeling/kuhn_poker_data_1w.json")
    print(f"Read {len(data)} items from file.")

    # split and save to train.parquet val.parquet test.parquet
    import pandas as pd
    df = pd.DataFrame(data)
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    # select n rows
    df = df[:64000]
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.2)
    test_size = len(df) - train_size - val_size

    train_df = df[:train_size]
    train_df.to_parquet("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/train_64k.parquet", index=False)

    val_df = df[train_size:train_size + val_size]
    val_df.to_parquet("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/val_64k.parquet", index=False)

    test_df = df[train_size + val_size:]
    # test_df.to_parquet("/home/cuisijia/llm_opponent_modeling/data/kuhn_poker/test_16k.parquet", index=False)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

if __name__ == "__main__":
    main()

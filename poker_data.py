
from datasets import load_dataset, load_from_disk

ds = load_from_disk("/home/cuisijia/llm_opponent_modeling/data/PokerBench")

print(ds["train"][0])
print(ds["test"][0])

'''
{'instruction': '\n\nYou are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\nThe small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\nIn this hand, your position is HJ, and your holding is [King of Diamond and Jack of Spade].\nBefore the flop, HJ raise 2.0 chips, and BB call. Assume that all other players that is not mentioned folded.\nThe flop comes King Of Spade, Seven Of Heart, and Two Of Diamond, then BB check, and HJ check.\nThe turn comes Jack Of Club, then BB check, HJ bet 3 chips, BB raise 10 chips, and HJ call.\nThe river comes Seven Of Club, then BB check.\n\n\nNow it is your turn to make a move.\nTo remind you, the current pot size is 24.0 chips, and your holding is [King of Diamond and Jack of Spade].\n\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\nYour optimal action is:', 'output': 'bet 18'}

-->

data = {
    "data_source": "RZ412/PokerBench",
    "prompt": [
        {
            "role": "user",
            "content": instruction,
        }
    ],
    "ability": "poker",
    "reward_model": {"style": "rule", "ground_truth": output},
    "extra_info": {
        "split": split,
        "index": idx,
        "answer": output,
        "question": instruction,
    },
}
'''
def make_map_fn(split):
    def process_fn(example, idx):
        instruction = example.pop("instruction")
        output = example.pop("output")
        
        data = {
            "data_source": "RZ412/PokerBench",
            "prompt": [
                {
                    "role": "user",
                    "content": instruction,
                }
            ],
            "ability": "poker",
            "reward_model": {"style": "rule", "ground_truth": output},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": output,
                "question": instruction,
            },
        }
        return data
    return process_fn

ds['train'] = ds['train'].map(make_map_fn("train"), with_indices=True)
ds['test'] = ds['test'].map(make_map_fn("test"), with_indices=True)

ds['train'].to_parquet("/home/cuisijia/llm_opponent_modeling/data/PokerBench/train.parquet")
ds['test'].to_parquet("/home/cuisijia/llm_opponent_modeling/data/PokerBench/test.parquet")


import pandas as pd
df = pd.read_parquet("/home/cuisijia/llm_opponent_modeling/data/PokerBench/train.parquet")
print(df.columns)

exit()

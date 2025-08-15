'''
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
        "index": len(data),
        "player_id": player_id,
        "player_card": player_card,
        "opponent_card": opponent_card,
        "action_history": action_history,
        "game_state": json.dumps(game_state),
    },
}
'''
import re

def compute_score(solution_str: str, ground_truth, extra_info):
    """
    Compute the reward score based on the solution string and ground truth.

    Args:
        solution_str (str): The generated solution string.
        ground_truth (str): The ground truth string to compare against.
        extra_info (dict): Additional information that may be needed for scoring.
    Returns:
        float: The computed reward score.
    """
    # solution_str format: <think> I will bet </think> <answer> [bet] </answer>
    # ground_truth format: [bet]

    solution_str = solution_str.strip()

    # tag reward
    reward_tag = (int(solution_str.count("<think>") > 0) + int(solution_str.count("<answer>") > 0) + \
                    int(solution_str.count("</think>") > 0) + int(solution_str.count("</answer>") > 0)) / 4.0
    
    # format reward
    # use regex to check the format: <think> I will bet </think> <answer> [bet] </answer>
    pattern = r"<think>.*</think>\s*<answer>\s*\[.*\]\s*</answer>"
    match = re.match(pattern, solution_str)
    reward_format = 1.0 if match else 0.0

    # correct reward
    # check if the action in solution_str matches the ground_truth
    action_match = re.search(r"<answer>\s*\[(.*?)\]\s*</answer>", solution_str)
    action = action_match.group(1).strip() if action_match else ""
    reward_correct = 1.0 if f"[{action}]" == ground_truth else 0.0

    return reward_correct + 0.1 * reward_format + 0.1 * reward_tag

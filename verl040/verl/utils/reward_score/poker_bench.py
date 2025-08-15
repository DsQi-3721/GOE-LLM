
def compute_score(solution_str, ground_truth):
    """
    Compute the reward score based on the solution string and ground truth.

    Args:
        solution_str (str): The solution string to evaluate.
        ground_truth (str): The ground truth string for comparison.
        data_source (str): The source of the data, default is "RZ412/PokerBench".

    Returns:
        float: The computed reward score.
    """
    # solution_str.contains(ground_truth)
    return 1.0 if ground_truth in solution_str else 0.0

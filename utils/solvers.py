import textarena as ta
from textarena.agents.basic_agents import HumanAgent
from utils.random_agent import RandomAgent, GtoAgent
# from utils.llm_agent import VLLMAgent

def run_one_episode(agents, env):
    """
    Run one episode of the Kuhn Poker game with the given agents and environment.
    """
    env.reset(num_players=len(agents))
    done = False 

    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)

    rewards, game_info = env.close()
    return rewards, game_info

if __name__ == "__main__":
    total_episodes = 1

    # Define agent combinations
    agent_combinations = [
        # (VLLMAgent, RandomAgent),
        # (RandomAgent, VLLMAgent),
        # (GtoAgent, VLLMAgent),
        # (VLLMAgent, GtoAgent),
        # (RandomAgent, RandomAgent),
        # (RandomAgent, GtoAgent),
        (GtoAgent, RandomAgent),
        # (GtoAgent, GtoAgent),
        # (VLLMAgent, VLLMAgent),
    ]

    # Initialize metrics to track the outcomes
    metrics = {
        "win_number": 0,
        "loss_number": 0,
        "draw_number": 0
    }

    # Loop through agent combinations
    for agent_1, agent_2 in agent_combinations:
        print(f"Testing combination: {str(agent_1())} vs {str(agent_2())}")
        
        # Reset episode-wise metrics for each agent combination
        episode_metrics = {
            "win_number": 0,
            "loss_number": 0,
            "draw_number": 0
        }

        for episode in range(total_episodes):
            print(f"Episode {episode + 1}/{total_episodes}")
            agents = {
                0: agent_1(),
                1: agent_2()
            }
            env = ta.make(env_id="KuhnPoker-v0", max_rounds=100)
            rewards, game_info = run_one_episode(agents, env)
            print(f"Rewards: {rewards}, Game Info: {game_info}")

            # rewards: {0: win_number, 1: win_number}
            if rewards[0] > rewards[1]:
                episode_metrics["win_number"] += 1
            elif rewards[0] < rewards[1]:
                episode_metrics["loss_number"] += 1
            else:
                episode_metrics["draw_number"] += 1

        # After all episodes for the combination, update overall metrics
        metrics["win_number"] += episode_metrics["win_number"]
        metrics["loss_number"] += episode_metrics["loss_number"]
        metrics["draw_number"] += episode_metrics["draw_number"]

        # Print results for the current combination
        print(f"Results for {str(agent_1())} vs {str(agent_2())}:")
        print(f"Wins: {episode_metrics['win_number']}, Losses: {episode_metrics['loss_number']}, Draws: {episode_metrics['draw_number']}")
    
    # Final aggregated results
    print(f"\nFinal aggregated results after {total_episodes * len(agent_combinations)} episodes:")
    print(f"Wins: {metrics['win_number']}")
    print(f"Losses: {metrics['loss_number']}")
    print(f"Draws: {metrics['draw_number']}")

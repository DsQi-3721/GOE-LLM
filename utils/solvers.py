import textarena as ta
from textarena.agents.basic_agents import HumanAgent
from utils.random_agent import RandomAgent, GtoAgent
# from utils.llm_agent import VLLMAgent

def run_gaming(agents, env):
    """
    Run one episode of the Kuhn Poker game with the given agents and environment.
    """
    env.reset(num_players=len(agents))
    done = False 

    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)
    
    print(f"Player 0 average utility: {env.state.game_state['player_chips'][0] / env.max_rounds} (GTO: {-1/18})", flush=True)
    win_num = env.player_0_wins
    rewards, game_info = env.close()
    return win_num, rewards, game_info

if __name__ == "__main__":
    total_rounds = 100000

    # Define agent combinations
    agent_combinations = [
        # (RandomAgent, RandomAgent),
        # (RandomAgent, GtoAgent),
        # (GtoAgent, RandomAgent),
        (GtoAgent, GtoAgent),
        # (VLLMAgent, RandomAgent),
        # (RandomAgent, VLLMAgent),
        # (GtoAgent, VLLMAgent),
        # (VLLMAgent, GtoAgent),
        # (VLLMAgent, VLLMAgent),
    ]

    # Loop through agent combinations
    for agent_1, agent_2 in agent_combinations:
        m1 = agent_1(alpha=1/3)
        m2 = agent_2(alpha=1/3)
        print(f"Testing combination: {str(m1)} vs {str(m2)}", flush=True)
        # Reset episode-wise metrics for each agent combination
        win_number = 0

        agents = {
            0: m1,
            1: m2,
        }
        env = ta.make(env_id="KuhnPoker-v0", max_rounds=total_rounds, changing_starting_player=False)
        win_number, rewards, game_info = run_gaming(agents, env)
        print(f"Results for {str(m1)} vs {str(m2)}:", flush=True)
        print(f"{str(m1)}: {win_number} --- {str(m2)}: {total_rounds - win_number}", flush=True)
        print(f"Rewards: {rewards}", flush=True)
        print(f"Game Info: {game_info}", flush=True)
        print("-" * 50, flush=True)
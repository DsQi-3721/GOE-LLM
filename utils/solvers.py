import textarena as ta
from utils.random_agent import RandomAgent

if __name__ == "__main__":
    
    agents = {
        0: RandomAgent(),
        1: RandomAgent()
    }

    env = ta.make(env_id="KuhnPoker-v0", max_rounds=1000)
    # env = ta.wrappers.SimpleRenderWrapper(env=env) #, render_mode="standard")

    env.reset(num_players=len(agents))

    # main game loop
    done = False 

    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)

    rewards, game_info = env.close()
    print(rewards)
    print(game_info)

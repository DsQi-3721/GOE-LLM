""" A minimal script showing how to run textarena locally """

import textarena as ta 

from textarena.core import Agent

from utils.solvers import VLLMAgent

agents = {
    0: ta.agents.HumanAgent(),
    1: ta.agents.HumanAgent(),
    # 1: ta.agents.OpenRouterAgent(model_name="google/gemini-2.0-flash-001"),
}

# initialize the environment
env = ta.make(env_id="KuhnPoker-v0")
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

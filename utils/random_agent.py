from textarena.core import Agent

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random

class RandomAgent(Agent):
    def __call__(self, observation: str) -> str:
        """
        Call the agent with an observation and return a random action.
        :param observation: The observation string from the environment.
        :return: A random action string.
        """
        logger.debug(f"Observation: {observation}")
        actions = observation.split("Your available actions are: ")[-1].strip().split(", ")
        action = random.choice(actions)
        logger.debug(f"Random action chosen: {action}")
        return action

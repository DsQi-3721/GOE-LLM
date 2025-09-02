from textarena.core import Agent
from utils.random_agent import RandomAgent, GtoAgent, describe_opponent
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicAgent(Agent):
    def __init__(self, interval_range: tuple, strategies: list, randomize_strategy: bool = False):
        """
        Initialize the DynamicAgent with a random interval range and a list of strategies.
        :param interval_range: A tuple (min_interval, max_interval) defining the range for random intervals.
        :param strategies: List of strategies.
        :param randomize_strategy: Whether to switch strategies in a random order.
        """
        super().__init__()
        self.min_interval, self.max_interval = interval_range
        self.strategies = strategies
        self.randomize_strategy = randomize_strategy
        self.current_strategy_index = 0 if not randomize_strategy else random.randint(0, len(self.strategies) - 1)
        self.current_step = 0
        self.next_switch_step = self._generate_random_interval()

    def _generate_random_interval(self) -> int:
        """
        Generate a random interval within the specified range.
        :return: A random interval.
        """
        return random.randint(self.min_interval, self.max_interval)

    def _switch_strategy(self):
        """
        Switch to the next strategy based on the randomize_strategy flag.
        """
        if self.randomize_strategy:
            # Randomly select a new strategy index
            self.current_strategy_index = random.randint(0, len(self.strategies) - 1)
        else:
            # Switch to the next strategy in a fixed order
            self.current_strategy_index = (self.current_strategy_index + 1) % len(self.strategies)

    def __call__(self, observation: str) -> str:
        """
        Call the agent with an observation and return an action based on the current strategy.
        :param observation: The observation string from the environment.
        :return: An action string.
        """
        # Switch strategy if the current step reaches the next switch step
        if self.current_step == self.next_switch_step:
            self._switch_strategy()
            logger.info("Switched to strategy: %s", str(self.strategies[self.current_strategy_index]))
            self.next_switch_step += self._generate_random_interval()

        # Delegate action to the current strategy
        action = self.strategies[self.current_strategy_index](observation)
        if observation.split('History: ', maxsplit=1)[-1].split('\n')[0].strip().count("->") == 1:
            pass # check -> bet: second action time of player0, same round
        else:
            self.current_step += 1
        return action

    def __str__(self):
        """
        Return the name of dynamic agent.
        """
        return f"DynamicAgent(interval_range=({self.min_interval}, {self.max_interval}), strategies={[str(s) for s in self.strategies]}, randomize_strategy={self.randomize_strategy})"

if __name__ == "__main__":
    # Example usage
    strategies = [
        RandomAgent(),
        GtoAgent(0.0),
        GtoAgent(1/6),
        GtoAgent(1/3),
        GtoAgent(1/2),
        GtoAgent(2/3),
        GtoAgent(5/6),
        GtoAgent(1),
    ]

    # Random order switching
    dynamic_agent = DynamicAgent(interval_range=(10, 50), strategies=strategies, randomize_strategy=True)
    print(describe_opponent(str(dynamic_agent)))
    for step in range(1000):
        observation = "Your card: 'K'\nHistory: Player 0: [check] -> Player 1: [bet]\nAvailable actions: [fold], [call]"
        action = dynamic_agent(observation)
        if step % 50 == 0:
            print(f"Step {step}: {action}")

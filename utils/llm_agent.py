from transformers import AutoTokenizer
from textarena.core import Agent
from textarena.agents.basic_agents import STANDARD_GAME_PROMPT
from vllm import LLM, SamplingParams

import logging
from threading import Lock
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 默认配置（可改） =====
DEFAULT_MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"
DEFAULT_TP_SIZE = 2

# ===== 懒加载单例（模块级共享）=====
__BACKEND_LOCK = Lock()
__SHARED_MODEL = None
__SHARED_TOKENIZER = None
__SHARED_CFG = {"model_path": None, "tp_size": None}

def _get_shared_backend(model_path: str = DEFAULT_MODEL_PATH, tensor_parallel_size: int = DEFAULT_TP_SIZE):
    """
    懒加载并返回（model, tokenizer）。只会在第一次调用时真正创建。
    之后调用如果配置一致，直接复用；如果配置不同，会给出日志提示并仍然复用已加载的实例。
    """
    global __SHARED_MODEL, __SHARED_TOKENIZER, __SHARED_CFG
    if __SHARED_MODEL is not None and __SHARED_TOKENIZER is not None:
        # 已经有共享实例
        if (__SHARED_CFG["model_path"] != model_path) or (__SHARED_CFG["tp_size"] != tensor_parallel_size):
            logger.warning(
                "Shared LLM already initialized with model=%s tp=%s, "
                "requested model=%s tp=%s. Reusing the existing one.",
                __SHARED_CFG['model_path'], __SHARED_CFG['tp_size'],
                model_path, tensor_parallel_size
            )
        return __SHARED_MODEL, __SHARED_TOKENIZER

    with __BACKEND_LOCK:
        if __SHARED_MODEL is None or __SHARED_TOKENIZER is None:
            logger.info("Initializing shared LLM: model=%s, tp=%d", model_path, tensor_parallel_size)
            __SHARED_MODEL = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
            __SHARED_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            __SHARED_CFG = {"model_path": model_path, "tp_size": tensor_parallel_size}
    return __SHARED_MODEL, __SHARED_TOKENIZER

# ===== 你已有的工具函数 =====
from utils.random_agent import clean_obs

def post_processing(response: str) -> str:
    if "\\boxed{" in response and "}" in response:
        return response.split("\\boxed{")[-1].split("}")[0].strip()
    elif "<answer>" in response and "</answer>" in response:
        return response.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        logger.debug("Response format is incorrect: %r", response)
        return response

def call_llm(messages: list[dict], sampling_params, model: LLM, tokenizer, thinking: bool = False) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    response = model.generate([text], sampling_params=sampling_params, use_tqdm=False)
    output = response[0].outputs[0].text
    return output.strip()

# ===== Agent 本体：首次实例化才加载，多个实例共享 =====
class VLLMAgent(Agent):
    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        *,
        # 可选：依赖注入/自定义加载配置
        model: LLM | None = None,
        tokenizer=None,
        model_path: str = DEFAULT_MODEL_PATH,
        tensor_parallel_size: int = DEFAULT_TP_SIZE,
    ):
        super().__init__()
        self.system_prompt = STANDARD_GAME_PROMPT
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        # 如果用户没手动注入，就拿共享单例（懒加载）
        if model is None or tokenizer is None:
            self.model, self.tokenizer = _get_shared_backend(model_path, tensor_parallel_size)
        else:
            self.model, self.tokenizer = model, tokenizer

        # 仅用于 __str__
        self._model_path = model_path
        self._tp_size = tensor_parallel_size

    def __call__(self, observation: str) -> str:
        logger.debug("%s Observation: %r", str(self), clean_obs(observation))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation},
        ]
        response = call_llm(messages, self.sampling_params, model=self.model, tokenizer=self.tokenizer, thinking=True)
        logger.debug("%s Action: %r", str(self), response)
        return post_processing(response)

    def __str__(self):
        return (f"VLLMAgent(model={self._model_path}, tp={self._tp_size}, "
                f"temp={self.sampling_params.temperature}, top_p={self.sampling_params.top_p}, "
                f"max_tokens={self.sampling_params.max_tokens})")


'''
You are an expert Kuhn Poker player.

[Game Rules]
- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest).
- Each player antes 1 chip and receives 1 card each round (note that the cards are dealt without replacement, so you cannot have the same card as your opponent).
- The player with the highest card wins the pot.

[Action Rules]
- [check]: Pass without betting (only if no bet is on the table)
- [bet]: Add 1 chip to the pot (only if no bet is on the table)
- [call]: Match an opponent's bet by adding 1 chip to the pot
- [fold]: Surrender your hand and let your opponent win the pot

[State]
You are Player 0 (first to act this round).
Your card: 'J'
History: Player 0: [check] -> Player 1: [bet]
Available actions: [fold], [call]

[Output Format]
``` plaintext
<think> Your thoughts and reasoning </think>
\\boxed{[ACTION]}
```

[Important Notes]
1. You must always include the <think> field to outline your reasoning.
2. Your final action [ACTION] must be one of the available actions, in \\boxed{[ACTION]} format.
'''
"""
LeducHoldem策略智能体设计
基于LeducHoldem的游戏特点设计不同类型的对手策略
"""

import random
import logging
import os
import re
from textarena.core import Agent
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def clean_obs(observation: str) -> str:
    """清理观察字符串，移除不必要的部分"""
    return observation.split('Available actions:')[-1].strip()

class PyCFRStrategyConverter:
    """PyCFR策略转换器"""
    
    def __init__(self):
        # pycfr动作映射
        self.PYCFR_FOLD = 0
        self.PYCFR_CALL = 1
        self.PYCFR_RAISE = 2
        
    def load_strategy_file(self, filepath: str) -> Dict[str, List[float]]:
        """
        加载pycfr策略文件
        
        Args:
            filepath: 策略文件路径
            
        Returns:
            Dict[str, List[float]]: 信息集到动作概率的映射
        """
        strategy = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(': ')
                if len(parts) != 2:
                    continue
                    
                infoset = parts[0]
                probs_str = parts[1].split()
                
                if len(probs_str) != 3:
                    continue
                    
                # 解析概率 [fold_prob, call_prob, raise_prob]
                probs = [float(p) for p in probs_str]
                strategy[infoset] = probs
                
        return strategy
    
    def parse_infoset(self, infoset: str) -> Dict[str, str]:
        """
        解析pycfr信息集格式
        
        Args:
            infoset: pycfr格式的信息集，如 "J:/", "JJ:/cc/cr:"
            
        Returns:
            Dict[str, str]: 解析后的信息
        """
        # 格式: "cards:bet_history:"
        parts = infoset.split(':')
        if len(parts) < 2:
            return {}
            
        cards = parts[0]
        bet_history = parts[1] if len(parts) > 1 else ""
        
        # 解析牌面
        if len(cards) == 1:
            # 单张牌，第一轮
            private_card = cards
            board_card = None
            round_num = 0
        elif len(cards) == 2:
            # 两张牌，第二轮
            private_card = cards[0]
            board_card = cards[1]
            round_num = 1
        else:
            return {}
            
        return {
            'private_card': private_card,
            'board_card': board_card,
            'bet_history': bet_history,
            'round': round_num
        }
    
    def get_game_state_from_history(self, bet_history: str, round_num: int) -> Dict[str, any]:
        """
        从下注历史推断游戏状态
        
        Args:
            bet_history: 下注历史
            round_num: 轮次
            
        Returns:
            Dict[str, any]: 游戏状态
        """
        current_bet = 0
        raises_this_round = 0
        
        # 分析下注历史
        for char in bet_history:
            if char == 'r':
                raises_this_round += 1
                if round_num == 0:
                    current_bet = 2  # 第一轮下注2
                else:
                    current_bet = 4  # 第二轮下注4
            elif char == 'c':
                if current_bet > 0:
                    # 有人下注后的跟注
                    pass
                else:
                    # 过牌
                    pass
                    
        return {
            'current_bet': current_bet,
            'raises_this_round': raises_this_round,
            'round': round_num
        }
    
    def convert_action_probs(self, pycfr_probs: List[float], game_state: Dict[str, any]) -> Dict[str, float]:
        """
        转换动作概率
        
        Args:
            pycfr_probs: pycfr格式的概率 [fold, call, raise]
            game_state: 游戏状态
            
        Returns:
            Dict[str, float]: textarena格式的动作概率
        """
        fold_prob, call_prob, raise_prob = pycfr_probs
        current_bet = game_state.get('current_bet', 0)
        
        if current_bet == 0:
            # 没有下注时，fold/call/raise -> fold/check/bet
            return {
                'fold': fold_prob,
                'check': call_prob,
                'bet': raise_prob
            }
        else:
            # 有下注时，fold/call/raise -> fold/call/raise
            return {
                'fold': fold_prob,
                'call': call_prob,
                'raise': raise_prob
            }
    
    def convert_strategy(self, strategy_file: str) -> Dict[str, Dict[str, float]]:
        """
        转换整个策略文件
        
        Args:
            strategy_file: pycfr策略文件路径
            
        Returns:
            Dict[str, Dict[str, float]]: 转换后的策略
        """
        pycfr_strategy = self.load_strategy_file(strategy_file)
        converted_strategy = {}
        
        for infoset, probs in pycfr_strategy.items():
            parsed = self.parse_infoset(infoset)
            if not parsed:
                continue
                
            game_state = self.get_game_state_from_history(
                parsed['bet_history'], 
                parsed['round']
            )
            
            action_probs = self.convert_action_probs(probs, game_state)
            
            # 构建textarena格式的信息集键
            key = self._build_textarena_key(parsed)
            converted_strategy[key] = action_probs
            
        return converted_strategy
    
    def _build_textarena_key(self, parsed_info: Dict[str, str]) -> str:
        """
        构建textarena格式的信息集键
        
        Args:
            parsed_info: 解析后的信息
            
        Returns:
            str: textarena格式的键
        """
        private_card = parsed_info['private_card']
        board_card = parsed_info.get('board_card')
        bet_history = parsed_info['bet_history']
        
        if board_card:
            return f"{private_card}{board_card}:{bet_history}"
        else:
            return f"{private_card}:{bet_history}"
    
    def get_action_probability(self, converted_strategy: Dict[str, Dict[str, float]], 
                             private_card: str, board_card: Optional[str], 
                             bet_history: str, action: str) -> float:
        """
        获取特定信息集和动作的概率
        
        Args:
            converted_strategy: 转换后的策略
            private_card: 手牌
            board_card: 公共牌（可选）
            bet_history: 下注历史
            action: 动作
            
        Returns:
            float: 动作概率
        """
        # 构建键
        if board_card:
            key = f"{private_card}{board_card}:{bet_history}"
        else:
            key = f"{private_card}:{bet_history}"
            
        if key not in converted_strategy:
            # 如果没有找到精确匹配，尝试模糊匹配
            return self._fuzzy_match_probability(converted_strategy, private_card, board_card, bet_history, action)
            
        action_probs = converted_strategy[key]
        return action_probs.get(action, 0.0)
    
    def _fuzzy_match_probability(self, converted_strategy: Dict[str, Dict[str, float]], 
                               private_card: str, board_card: Optional[str], 
                               bet_history: str, action: str) -> float:
        """
        模糊匹配概率（当没有精确匹配时）
        
        Args:
            converted_strategy: 转换后的策略
            private_card: 手牌
            board_card: 公共牌（可选）
            bet_history: 下注历史
            action: 动作
            
        Returns:
            float: 动作概率
        """
        # 简单的启发式匹配
        if action == 'fold':
            return 0.1  # 默认fold概率
        elif action in ['check', 'call']:
            return 0.7  # 默认check/call概率
        elif action in ['bet', 'raise']:
            return 0.2  # 默认bet/raise概率
        else:
            return 0.0

class LeducRandomAgent(Agent):
    """随机策略智能体"""
    def __call__(self, observation: str) -> str:
        actions = observation.split("Available actions: ")[-1].split('\n')[0].strip().split(", ")
        action = random.choice(actions)
        logger.debug("RandomAgent Action: %r", action.strip("'"))
        return action

    def call_parallel(self, observation: str, n: int) -> list:
        """批量生成动作以加速评测"""
        return [self.__call__(observation) for _ in range(n)]

    def __str__(self):
        return "LeducRandomAgent"

class LeducGTOAgent(Agent):
    """基于PyCFR策略的GTO智能体"""
    
    def __init__(self, strategy_file: str = None, player_id: int = 0, precision: float = 1):
        """
        初始化GTO智能体
        
        Args:
            strategy_file: PyCFR策略文件路径
            player_id: 玩家ID (0或1)
            precision: 策略精度（用于随机化）
        """
        super().__init__()
        self.precision = precision
        self.player_id = player_id
        self.converter = PyCFRStrategyConverter()
        self.strategy = {}
        
        # 默认策略文件路径
        if strategy_file is None:
            # 使用相对路径，根据玩家ID选择策略文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            strategy_file = os.path.join(current_dir, f"leduc_standard_cfr_player{player_id}.strat")
            
        self.strategy_file = strategy_file
        self._load_strategy()
        self.agent_name = f"LeducGTOAgent(player{player_id})"
        
    def _load_strategy(self):
        """加载PyCFR策略"""
        try:
            if os.path.exists(self.strategy_file):
                self.strategy = self.converter.convert_strategy(self.strategy_file)
                logger.info(f"成功加载策略文件: {self.strategy_file}, 包含 {len(self.strategy)} 个信息集")
            else:
                logger.warning(f"策略文件不存在: {self.strategy_file}")
                self.strategy = {}
        except Exception as e:
            logger.error(f"加载策略文件失败: {e}")
            self.strategy = {}
    
    def _parse_observation(self, observation: str) -> Dict[str, str]:
        """
        解析观察信息
        
        Args:
            observation: 观察字符串
            
        Returns:
            Dict[str, str]: 解析后的游戏信息
        """
        info = {}
        
        # 提取手牌
        card_match = re.search(r"Your card: '([JQK])'", observation)
        if card_match:
            info['private_card'] = card_match.group(1)
            
        # 提取公共牌
        board_match = re.search(r"Board card: '([JQK])'", observation)
        if board_match:
            info['board_card'] = board_match.group(1)
        else:
            info['board_card'] = None
            
        # 提取历史
        history_match = re.search(r"History: (.*?)(?:\n|$)", observation)
        if history_match:
            history_str = history_match.group(1).strip()
            info['bet_history'] = self._parse_history_string(history_str)
        else:
            info['bet_history'] = ""
            
        # 提取可用动作
        actions_match = re.search(r"Available actions: (.*?)(?:\n|$)", observation)
        if actions_match:
            actions_str = actions_match.group(1).strip()
            info['available_actions'] = [action.strip("[]") for action in actions_str.split(", ")]
        else:
            info['available_actions'] = []
            
        return info
    
    def _parse_history_string(self, history_str: str) -> str:
        """
        解析历史字符串为下注历史
        
        Args:
            history_str: 历史字符串，如 "Player 0: [check] -> Player 1: [bet]"
            
        Returns:
            str: 下注历史，如 "cr"
        """
        if not history_str:
            return ""
            
        bet_history = ""
        actions = re.findall(r'\[(check|bet|call|raise|fold)\]', history_str)
        
        for action in actions:
            if action == 'check':
                bet_history += 'c'  # check
            elif action == 'bet':
                bet_history += 'r'  # bet/raise
            elif action == 'call':
                bet_history += 'c'  # call (在pycfr中call和check都用c表示)
            elif action == 'raise':
                bet_history += 'r'  # raise
            elif action == 'fold':
                bet_history += 'f'  # fold
                
        return bet_history
    
    def _get_action_probability(self, private_card: str, board_card: Optional[str], 
                               bet_history: str, action: str) -> float:
        """
        获取动作概率
        
        Args:
            private_card: 手牌
            board_card: 公共牌
            bet_history: 下注历史
            action: 动作
            
        Returns:
            float: 动作概率
        """
        return self.converter.get_action_probability(
            self.strategy, private_card, board_card, bet_history, action
        )
    
    def _normalize_probabilities(self, action_probs: Dict[str, float], available_actions: List[str]) -> Dict[str, float]:
        """
        归一化动作概率
        
        Args:
            action_probs: 原始动作概率分布
            available_actions: 可用动作列表
            
        Returns:
            Dict[str, float]: 归一化后的概率分布
        """
        # 过滤可用动作并确保概率非负
        valid_probs = {}
        for action in available_actions:
            if action in action_probs:
                valid_probs[action] = max(0.0, action_probs[action])
            else:
                valid_probs[action] = 0.0
        
        # 计算总概率
        total_prob = sum(valid_probs.values())
        
        if total_prob == 0:
            # 如果没有有效概率，均匀分布
            return {action: 1.0 / len(available_actions) for action in available_actions}
        
        # 归一化
        normalized_probs = {action: prob / total_prob for action, prob in valid_probs.items()}
        
        # 添加随机性
        if self.precision < 1.0:
            uniform_prob = 1.0 / len(available_actions)
            for action in available_actions:
                normalized_probs[action] = (
                    self.precision * normalized_probs.get(action, 0) + 
                    (1 - self.precision) * uniform_prob
                )
        
        return normalized_probs
    
    def _sample_action(self, action_probs: Dict[str, float], available_actions: List[str]) -> str:
        """
        根据概率分布采样动作
        
        Args:
            action_probs: 动作概率分布
            available_actions: 可用动作列表
            
        Returns:
            str: 选择的动作
        """
        # 归一化概率
        normalized_probs = self._normalize_probabilities(action_probs, available_actions)
        
        # 采样
        rand = random.random()
        cumulative = 0.0
        
        for action, prob in normalized_probs.items():
            cumulative += prob
            if rand <= cumulative:
                return f"[{action}]"
                
        # 备用：返回最后一个动作
        return f"[{available_actions[-1]}]"
    
    def __call__(self, observation: str) -> str:
        """
        根据观察选择动作
        
        Args:
            observation: 观察字符串
            
        Returns:
            str: 选择的动作
        """
        try:
            # 解析观察
            info = self._parse_observation(observation)
            
            if not info.get('private_card'):
                logger.warning("无法解析手牌信息")
                return "[fold]"
                
            # 获取动作概率
            action_probs = {}
            for action in info['available_actions']:
                prob = self._get_action_probability(
                    info['private_card'],
                    info['board_card'],
                    info['bet_history'],
                    action
                )
                action_probs[action] = prob
            
            # 采样动作
            chosen_action = self._sample_action(action_probs, info['available_actions'])
            
            logger.debug(f"GTO Agent选择动作: {chosen_action}")
            return chosen_action
            
        except Exception as e:
            logger.error(f"GTO Agent选择动作时出错: {e}")
            # 备用：随机选择
            available_actions = re.findall(r'\[([^\]]+)\]', observation)
            if available_actions:
                return f"[{random.choice(available_actions)}]"
            else:
                return "[fold]"
    
    def call_parallel(self, observation: str, n: int) -> list:
        """批量生成动作以加速评测"""
        return [self.__call__(observation) for _ in range(n)]

    def __str__(self):
        return self.agent_name

class LeducTightAgent(LeducGTOAgent):
    """
    紧手策略：极度保守，很少诈唬，主要玩强牌
    【修改】使用指数衰减算法大幅降低弱牌激进行为，显著增强策略特征
    """
    def __init__(self, tightness: float = 0.8, player_id: int = 0):
        super().__init__(player_id=player_id, precision=0.95)
        self.tightness = tightness
        self.agent_name = f"LeducTightAgent({tightness:.2f},player{player_id})"
    
    def _get_action_probability(self, private_card: str, board_card: Optional[str], 
                               bet_history: str, action: str) -> float:
        """
        【修改】极度保守的概率调整算法
        - 使用指数衰减大幅降低弱牌的激进行为
        - 显著增加弃牌倾向
        - 只有顶级牌才会主动下注
        """
        base_prob = super()._get_action_probability(private_card, board_card, bet_history, action)
        
        # 根据牌力调整概率
        card_strength = self._get_card_strength(private_card, board_card)
        
        if action in ['bet', 'raise']:
            # 【修改】指数衰减算法：弱牌几乎不下注
            if card_strength < 0.8:  # 非顶级牌力
                # 指数衰减：tightness=0.9时，弱牌下注概率降到原来的1%
                decay_factor = (1 - self.tightness) ** 3
                return max(0.01, base_prob * decay_factor)
            else:  # 顶级牌力(K对子)
                # 强牌稍微保守
                return max(0.1, base_prob * (1 - self.tightness * 0.2))
                
        elif action == 'call':
            # 【修改】大幅降低跟注频率，特别是弱牌
            if card_strength < 0.6:  # 中等以下牌力
                # 弱牌跟注概率大幅下降
                return max(0.05, base_prob * (1 - self.tightness) ** 2)
            else:  # 强牌
                return max(0.2, base_prob * (1 - self.tightness * 0.4))
                
        elif action == 'fold':
            # 【修改】显著增加弃牌频率，使用平方增长
            fold_boost = self.tightness ** 0.5  # 平方根增长，更平滑
            return min(0.95, base_prob + fold_boost * 0.6)
            
        elif action == 'check':
            # 【修改】增加过牌频率（被动行为）
            return min(0.9, base_prob + self.tightness * 0.3)
        
        return base_prob
    
    def _get_card_strength(self, private_card: str, board_card: Optional[str]) -> float:
        """计算牌力强度"""
        if board_card and private_card == board_card:
            return 1.0  # 对子
        elif private_card == 'K':
            return 0.8
        elif private_card == 'Q':
            return 0.6
        elif private_card == 'J':
            return 0.4
        else:
            return 0.0

class LeducLooseAgent(LeducGTOAgent):
    """
    松手策略：极度松散，经常诈唬，玩很多牌
    【修改】使用对数增长算法大幅增加弱牌激进行为，显著降低弃牌倾向
    """
    def __init__(self, looseness: float = 0.8, player_id: int = 0):
        super().__init__(player_id=player_id, precision=0.95)
        self.looseness = looseness
        self.agent_name = f"LeducLooseAgent({looseness:.2f},player{player_id})"
    
    def _get_action_probability(self, private_card: str, board_card: Optional[str], 
                               bet_history: str, action: str) -> float:
        """
        【修改】极度松散的概率调整算法
        - 使用对数增长大幅增加弱牌的诈唬行为
        - 显著降低弃牌倾向（几乎不弃牌）
        - 所有牌力都倾向于激进行为
        """
        base_prob = super()._get_action_probability(private_card, board_card, bet_history, action)
        
        # 计算牌力
        card_strength = self._get_card_strength(private_card, board_card)
        
        if action in ['bet', 'raise']:
            # 【修改】对数增长算法：弱牌大幅增加诈唬
            if card_strength < 0.6:  # 弱牌和中等牌
                # 对数增长：looseness=0.9时，弱牌下注概率可达90%+
                import math
                boost_factor = 1 + self.looseness * math.log(1 + self.looseness * 4)
                return min(0.95, base_prob * boost_factor + self.looseness * 0.4)
            else:  # 强牌
                # 强牌也更激进
                return min(0.98, base_prob + self.looseness * 0.2)
                
        elif action == 'call':
            # 【修改】大幅增加跟注频率（松手玩家特征）
            # 几乎任何牌都愿意跟注
            call_boost = self.looseness ** 0.7  # 0.7次方增长
            return min(0.9, base_prob + call_boost * 0.5)
                
        elif action == 'fold':
            # 【修改】极度降低弃牌频率（松手核心特征）
            # 使用指数衰减让弃牌概率接近0
            fold_reduction = self.looseness ** 2  # 平方衰减
            return max(0.02, base_prob * (1 - fold_reduction * 0.8))
            
        elif action == 'check':
            # 【修改】降低过牌频率（倾向于主动行为）
            return max(0.1, base_prob * (1 - self.looseness * 0.3))
        
        return base_prob
    
    def _get_card_strength(self, private_card: str, board_card: Optional[str]) -> float:
        """计算牌力强度"""
        if board_card and private_card == board_card:
            return 1.0  # 对子
        elif private_card == 'K':
            return 0.8
        elif private_card == 'Q':
            return 0.6
        elif private_card == 'J':
            return 0.4
        else:
            return 0.0

class LeducAggressiveAgent(LeducGTOAgent):
    """
    激进策略：极度激进，经常下注和加注，施加压力
    【修改】使用双曲函数算法大幅增加所有激进行为，显著区别于其他策略
    """
    def __init__(self, aggression: float = 0.8, player_id: int = 0):
        super().__init__(player_id=player_id, precision=0.95)
        self.aggression = aggression
        self.agent_name = f"LeducAggressiveAgent({aggression:.2f},player{player_id})"
    
    def _get_action_probability(self, private_card: str, board_card: Optional[str], 
                               bet_history: str, action: str) -> float:
        """
        【修改】极度激进的概率调整算法
        - 使用双曲函数大幅增加bet/raise行为
        - 显著降低被动行为(check/fold)
        - 所有牌力都倾向于主动施压
        """
        base_prob = super()._get_action_probability(private_card, board_card, bet_history, action)
        
        # 计算牌力
        card_strength = self._get_card_strength(private_card, board_card)
        
        if action in ['bet', 'raise']:
            # 【修改】双曲函数增长：激进下注/加注
            import math
            # 使用tanh函数创造非线性增长
            aggression_boost = math.tanh(self.aggression * 2) * 0.7  # 最大增加70%
            
            if card_strength < 0.4:  # 弱牌：激进诈唬
                return min(0.9, base_prob + aggression_boost + self.aggression * 0.3)
            elif card_strength < 0.8:  # 中等牌：极度激进
                return min(0.95, base_prob + aggression_boost + self.aggression * 0.4)
            else:  # 强牌：最大化价值
                return min(0.98, base_prob + aggression_boost + self.aggression * 0.2)
                
        elif action == 'call':
            # 【修改】适度增加跟注（激进但不鲁莽）
            if card_strength > 0.3:  # 有一定牌力才跟注
                return min(0.8, base_prob + self.aggression * 0.3)
            else:
                # 弱牌更倾向于主动下注而非跟注
                return max(0.1, base_prob * (1 - self.aggression * 0.2))
                
        elif action == 'fold':
            # 【修改】极度降低弃牌频率（激进核心特征）
            # 激进玩家几乎不弃牌，宁可诈唬
            fold_penalty = self.aggression ** 1.5  # 1.5次方惩罚
            return max(0.05, base_prob * (1 - fold_penalty * 0.7))
            
        elif action == 'check':
            # 【修改】显著降低过牌频率（倾向于主动下注）
            check_penalty = self.aggression ** 0.8  # 0.8次方惩罚
            return max(0.1, base_prob * (1 - check_penalty * 0.5))
        
        return base_prob
    
    def _get_card_strength(self, private_card: str, board_card: Optional[str]) -> float:
        """计算牌力强度"""
        if board_card and private_card == board_card:
            return 1.0  # 对子
        elif private_card == 'K':
            return 0.8
        elif private_card == 'Q':
            return 0.6
        elif private_card == 'J':
            return 0.4
        else:
            return 0.0

class LeducPassiveAgent(LeducGTOAgent):
    """
    被动策略：极度被动，很少主动下注，主要跟注和过牌
    【修改】使用Sigmoid函数算法大幅降低主动行为，显著增强被动特征
    """
    def __init__(self, passiveness: float = 0.8, player_id: int = 0):
        super().__init__(player_id=player_id, precision=0.95)
        self.passiveness = passiveness
        self.agent_name = f"LeducPassiveAgent({passiveness:.2f},player{player_id})"
    
    def _get_action_probability(self, private_card: str, board_card: Optional[str], 
                               bet_history: str, action: str) -> float:
        """
        【修改】极度被动的概率调整算法
        - 使用Sigmoid函数大幅降低主动行为(bet/raise)
        - 显著增加被动行为(check/call)
        - 即使强牌也倾向于被动游戏
        """
        base_prob = super()._get_action_probability(private_card, board_card, bet_history, action)
        
        # 计算牌力
        card_strength = self._get_card_strength(private_card, board_card)
        
        if action in ['bet', 'raise']:
            # 【修改】Sigmoid衰减：极度降低主动下注
            import math
            # 使用sigmoid函数创造平滑的衰减曲线
            sigmoid_factor = 1 / (1 + math.exp(self.passiveness * 5))  # 5倍放大效果
            
            if card_strength < 0.9:  # 非完美牌力
                # 大幅降低主动下注，即使中等牌力也很少下注
                return max(0.02, base_prob * sigmoid_factor * 0.3)
            else:  # 完美牌力(对子)
                # 即使最强牌也相对被动
                return max(0.1, base_prob * sigmoid_factor * 0.6)
                
        elif action == 'call':
            # 【修改】显著增加跟注频率（被动核心特征）
            # 被动玩家喜欢跟注而不主动下注
            call_boost = self.passiveness ** 0.5  # 平方根增长，更平滑
            if card_strength > 0.2:  # 有基本牌力就愿意跟注
                return min(0.85, base_prob + call_boost * 0.6)
            else:  # 极弱牌也适度跟注（被动特征）
                return min(0.4, base_prob + call_boost * 0.3)
                
        elif action == 'check':
            # 【修改】大幅增加过牌频率（被动核心特征）
            # 被动玩家最喜欢过牌观望
            check_boost = self.passiveness ** 0.3  # 0.3次方增长，强化效果
            return min(0.9, base_prob + check_boost * 0.5)
            
        elif action == 'fold':
            # 【修改】适度增加弃牌频率（面对压力时容易退缩）
            if card_strength < 0.5:  # 弱牌面对压力容易弃牌
                return min(0.8, base_prob + self.passiveness * 0.2)
            else:  # 强牌相对坚持
                return base_prob
        
        return base_prob
    
    def _get_card_strength(self, private_card: str, board_card: Optional[str]) -> float:
        """计算牌力强度"""
        if board_card and private_card == board_card:
            return 1.0  # 对子
        elif private_card == 'K':
            return 0.8
        elif private_card == 'Q':
            return 0.6
        elif private_card == 'J':
            return 0.4
        else:
            return 0.0

def describe_leduc_opponent(agent_name: str) -> str:
    """
    将LeducHoldem智能体名称转换为自然语言描述
    用于LLM提示条件
    """
    agent_name = agent_name.strip()

    if agent_name == "LeducRandomAgent":
        return ("The opponent plays completely randomly in Leduc Hold'em, without any consistent strategy. "
                "Their actions are unpredictable and not based on card strength or position.")

    if agent_name.startswith("LeducTightAgent"):
        return ("The opponent plays a tight strategy in Leduc Hold'em. "
                "They rarely bluff and mostly play strong hands (K, Q). "
                "They can be exploited by bluffing more often, especially with J.")

    if agent_name.startswith("LeducLooseAgent"):
        return ("The opponent plays a loose strategy in Leduc Hold'em. "
                "They play many hands and bluff frequently. "
                "They can be exploited by calling more often with medium-strength hands.")

    if agent_name.startswith("LeducAggressiveAgent"):
        return ("The opponent plays an aggressive strategy in Leduc Hold'em. "
                "They bet and raise frequently, putting pressure on opponents. "
                "They can be exploited by calling wider and playing more defensively.")

    if agent_name.startswith("LeducPassiveAgent"):
        return ("The opponent plays a passive strategy in Leduc Hold'em. "
                "They rarely bet or raise, mostly calling and checking. "
                "They can be exploited by betting more often and applying pressure.")

    if agent_name.startswith("LeducGTOAgent"):
        return ("The opponent attempts to play according to equilibrium strategies in Leduc Hold'em, "
                "balancing value bets and bluffs across different betting rounds.")

    # 默认描述
    return "The opponent's strategy in Leduc Hold'em is unknown or unusual."

if __name__ == "__main__":
    # 测试智能体
    agents = [
        LeducRandomAgent(),
        LeducGTOAgent(),
        LeducTightAgent(tightness=0.8),
        LeducLooseAgent(looseness=0.8),
        LeducAggressiveAgent(aggression=0.8),
        LeducPassiveAgent(passiveness=0.8)
    ]
    
    for agent in agents:
        print(f"{agent}: {describe_leduc_opponent(str(agent))}")

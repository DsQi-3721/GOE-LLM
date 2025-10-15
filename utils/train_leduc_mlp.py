from utils.leduc_agents import (
    LeducRandomAgent, LeducGTOAgent, LeducTightAgent, LeducLooseAgent, 
    LeducAggressiveAgent, LeducPassiveAgent, describe_leduc_opponent
)
import textarena as ta
import json
import os
import re
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

def get_agent_class(agent_name):
    """将智能体名称映射到类别ID"""
    if "LeducRandomAgent" in agent_name:
        return 0
    elif "LeducTightAgent" in agent_name:
        return 1
    elif "LeducLooseAgent" in agent_name:
        return 2
    elif "LeducAggressiveAgent" in agent_name:
        return 3
    elif "LeducPassiveAgent" in agent_name:
        return 4
    elif "LeducGTOAgent" in agent_name:
        return 5
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")

name_mapping = {
    0: "LeducRandomAgent", 
    1: "LeducTightAgent", 
    2: "LeducLooseAgent", 
    3: "LeducAggressiveAgent", 
    4: "LeducPassiveAgent",
    5: "LeducGTOAgent"
}

class LeducDecisionHistory:
    """Leduc Holdem决策历史记录"""
    def __init__(self):
        self.tree = {
            "as_player_0": {
                "J": {},
                "Q": {},
                "K": {},
            },
            "as_player_1": {
                "J": {},
                "Q": {},
                "K": {},
            }
        }

def run_leduc_eval(eval_agent, opponent_agent, max_rounds=100):
    """
    运行Leduc Holdem评估
    
    Args:
        eval_agent: 被评估的智能体
        opponent_agent: 对手智能体
        max_rounds: 总轮数
        
    Returns:
        dict: 评估结果和决策历史
    """
    dh = LeducDecisionHistory()
    wrong_action = {0: 0, 1: 0}
    
    def _record(env, player_id, action):
        """记录决策历史"""
        match = re.compile(r"\[(check|bet|call|raise|fold)\]", re.IGNORECASE).search(action.strip())
        if not match: 
            wrong_action[player_id] += 1
        else:
            action = f'[{match.group(1).lower()}]'
            if action in ["[check]", "[bet]", "[call]", "[raise]", "[fold]"]:
                # 获取手牌
                card = env._rank_to_str(env.state.game_state["player_cards"][player_id])
                # 获取历史
                his = '->'.join(env.history) if env.history else ""
                
                # 记录决策
                player_key = f"as_player_{player_id}"
                if his not in dh.tree[player_key][card]:
                    dh.tree[player_key][card][his] = {}
                dh.tree[player_key][card][his][action.lower()] = \
                    dh.tree[player_key][card][his].get(action.lower(), 0) + 1
            else:
                wrong_action[player_id] += 1

    # eval_agent作为玩家0，opponent作为玩家1
    agents = {0: eval_agent, 1: opponent_agent}
    env0 = ta.make(env_id="LeducHoldem-v0", max_rounds=max_rounds, changing_starting_player=False)
    env0.reset(num_players=len(agents))
    done = False
    
    while not done:
        player_id, observation = env0.get_observation()
        action = agents[player_id](observation)
        
        if player_id == 0:  # 记录eval_agent的决策
            _record(env0, player_id, action)
        
        done, step_info = env0.step(action=action)
    
    # eval_agent作为玩家1，opponent作为玩家0
    agents = {1: eval_agent, 0: opponent_agent}
    env1 = ta.make(env_id="LeducHoldem-v0", max_rounds=max_rounds, changing_starting_player=False)
    env1.reset(num_players=len(agents))
    done = False
    
    while not done:
        player_id, observation = env1.get_observation()
        action = agents[player_id](observation)
        
        if player_id == 1:  # 记录eval_agent的决策
            _record(env1, player_id, action)
        
        done, step_info = env1.step(action=action)
    
    # 安全地获取游戏状态信息
    try:
        player_0_bank = env0.state.game_state.get("player_bank", {}).get(0, 0)
        player_1_bank = env1.state.game_state.get("player_bank", {}).get(1, 0)
    except Exception as e:
        print(f"Error accessing player_bank: {e}")
        print(f"env0 game_state keys: {list(env0.state.game_state.keys())}")
        print(f"env1 game_state keys: {list(env1.state.game_state.keys())}")
        player_0_bank = player_1_bank = 0
    
    eval_agent_metrics = {
        "agent": str(eval_agent),
        "opponent": str(opponent_agent),
        "max_rounds": max_rounds,
        "as_player_0": {
            "win_num": getattr(env0, 'player_0_wins', 0),
            "win_chips": player_0_bank,
            "avg_win_chips": player_0_bank / max_rounds if max_rounds > 0 else 0,
        },
        "as_player_1": {
            "win_num": max_rounds - getattr(env1, 'player_0_wins', 0),
            "win_chips": player_1_bank,
            "avg_win_chips": player_1_bank / max_rounds if max_rounds > 0 else 0,
        },
        "decision_history": dh.tree,
        "wrong_action": wrong_action,
    }
    
    env0.close()
    env1.close()
    return eval_agent_metrics

def processing_leduc_decision_history(dh):
    """
    处理Leduc Holdem决策历史，转换为特征向量
    
    增强版特征提取，更好地捕获策略差异：
    - 分别提取不同手牌强度的行为模式
    - 计算激进度、保守度等策略指标
    - 区分第一轮和第二轮的行为差异
    
    特征向量结构（60维）：
    - 基础行为模式: 48维（原有）
    - 策略指标: 12维（新增）
      - 整体激进度（bet/raise频率）
      - 整体保守度（fold频率）
      - 诈唬倾向（弱牌下注频率）
      - 价值下注倾向（强牌下注频率）
    """
    x = [0.0 for _ in range(60)]  # 增加到60维特征向量
    
    try:
        idx = 0
        
        # 作为玩家0的决策模式 (24维)
        for card in ['J', 'Q', 'K']:
            card_data = dh.get('as_player_0', {}).get(card, {})
            
            # 第一轮：初始决策（check vs bet）2维
            initial_actions = card_data.get('', {})
            total_initial = sum(initial_actions.values())
            if total_initial > 0:
                x[idx] = initial_actions.get('[check]', 0) / total_initial
                x[idx+1] = initial_actions.get('[bet]', 0) / total_initial
            idx += 2
            
            # 第一轮：面对下注的反应（call vs raise vs fold）3维
            facing_bet_keys = [k for k in card_data.keys() if 'check' in k and 'bet' in k]
            if facing_bet_keys:
                facing_bet_actions = card_data[facing_bet_keys[0]]
                total_facing_bet = sum(facing_bet_actions.values())
                if total_facing_bet > 0:
                    x[idx] = facing_bet_actions.get('[call]', 0) / total_facing_bet
                    x[idx+1] = facing_bet_actions.get('[raise]', 0) / total_facing_bet
                    x[idx+2] = facing_bet_actions.get('[fold]', 0) / total_facing_bet
            idx += 3
            
            # 第二轮决策模式聚合 3维
            second_round_actions = {'[check]': 0, '[bet]': 0, '[call]': 0}
            for key, actions in card_data.items():
                if '->' in key and len(key.split('->')) > 2:  # 第二轮的标志
                    for action, count in actions.items():
                        if action in second_round_actions:
                            second_round_actions[action] += count
            
            total_second = sum(second_round_actions.values())
            if total_second > 0:
                x[idx] = second_round_actions['[check]'] / total_second
                x[idx+1] = second_round_actions['[bet]'] / total_second
                x[idx+2] = second_round_actions['[call]'] / total_second
            idx += 3
        
        # 作为玩家1的决策模式 (24维)
        for card in ['J', 'Q', 'K']:
            card_data = dh.get('as_player_1', {}).get(card, {})
            
            # 面对check的决策 (check vs bet) 2维
            facing_check_keys = [k for k in card_data.keys() if 'check' in k and 'bet' not in k]
            if facing_check_keys:
                facing_check_actions = card_data[facing_check_keys[0]]
                total_facing_check = sum(facing_check_actions.values())
                if total_facing_check > 0:
                    x[idx] = facing_check_actions.get('[check]', 0) / total_facing_check
                    x[idx+1] = facing_check_actions.get('[bet]', 0) / total_facing_check
            idx += 2
            
            # 面对bet的决策 (call vs raise vs fold) 3维
            facing_bet_keys = [k for k in card_data.keys() if 'bet' in k]
            if facing_bet_keys:
                facing_bet_actions = card_data[facing_bet_keys[0]]
                total_facing_bet = sum(facing_bet_actions.values())
                if total_facing_bet > 0:
                    x[idx] = facing_bet_actions.get('[call]', 0) / total_facing_bet
                    x[idx+1] = facing_bet_actions.get('[raise]', 0) / total_facing_bet  
                    x[idx+2] = facing_bet_actions.get('[fold]', 0) / total_facing_bet
            idx += 3
            
            # 第二轮反应模式 3维
            second_round_actions = {'[check]': 0, '[bet]': 0, '[call]': 0}
            for key, actions in card_data.items():
                if '->' in key and len(key.split('->')) > 2:  # 第二轮
                    for action, count in actions.items():
                        if action in second_round_actions:
                            second_round_actions[action] += count
            
            total_second = sum(second_round_actions.values())
            if total_second > 0:
                x[idx] = second_round_actions['[check]'] / total_second
                x[idx+1] = second_round_actions['[bet]'] / total_second
                x[idx+2] = second_round_actions['[call]'] / total_second
            idx += 3
        
        # 新增：策略指标特征 (12维)
        # 计算整体行为统计
        total_actions = {'[check]': 0, '[bet]': 0, '[call]': 0, '[raise]': 0, '[fold]': 0}
        card_based_actions = {'J': {'[bet]': 0, '[raise]': 0, 'total': 0}, 
                             'Q': {'[bet]': 0, '[raise]': 0, 'total': 0}, 
                             'K': {'[bet]': 0, '[raise]': 0, 'total': 0}}
        
        for player in ['as_player_0', 'as_player_1']:
            for card in ['J', 'Q', 'K']:
                card_data = dh.get(player, {}).get(card, {})
                for situation, actions in card_data.items():
                    for action, count in actions.items():
                        if action in total_actions:
                            total_actions[action] += count
                        if action in ['[bet]', '[raise]']:
                            card_based_actions[card][action] += count
                        card_based_actions[card]['total'] += count
        
        total_decisions = sum(total_actions.values())
        if total_decisions > 0:
            # 整体激进度 (bet + raise频率) - 2维
            x[idx] = (total_actions['[bet]'] + total_actions['[raise]']) / total_decisions
            x[idx+1] = total_actions['[check]'] / total_decisions
            idx += 2
            
            # 整体保守度 (fold频率 vs call频率) - 2维  
            x[idx] = total_actions['[fold]'] / total_decisions
            x[idx+1] = total_actions['[call]'] / total_decisions
            idx += 2
            
            # 按牌力的行为差异 - 6维 (每种牌2维)
            for card in ['J', 'Q', 'K']:
                if card_based_actions[card]['total'] > 0:
                    # 该牌力下的激进度
                    aggressive_actions = card_based_actions[card]['[bet]'] + card_based_actions[card]['[raise]']
                    x[idx] = aggressive_actions / card_based_actions[card]['total']
                else:
                    x[idx] = 0.0
                idx += 1
                
                # 该牌力下的决策数量占比（反映该牌力的游戏频率）
                x[idx] = card_based_actions[card]['total'] / total_decisions if total_decisions > 0 else 0.0
                idx += 1
            
            # 策略一致性指标 - 2维
            # 不同牌力间的行为一致性（标准差）
            card_aggression = []
            for card in ['J', 'Q', 'K']:
                if card_based_actions[card]['total'] > 0:
                    aggressive_actions = card_based_actions[card]['[bet]'] + card_based_actions[card]['[raise]']
                    card_aggression.append(aggressive_actions / card_based_actions[card]['total'])
                else:
                    card_aggression.append(0.0)
            
            if len(card_aggression) > 1:
                x[idx] = np.std(card_aggression)  # 行为一致性（低标准差=一致）
            idx += 1
            x[idx] = np.mean(card_aggression)  # 平均激进度
            idx += 1
        else:
            idx += 12  # 跳过策略指标部分
            
    except Exception as e:
        print(f"Error in processing Leduc decision history: {dh}")
        print(f"Error details: {e}")
        # 返回零向量作为备选
        x = [0.0 for _ in range(60)]
    
    return x

def generate_leduc_train_data(eval_agent, max_rounds, data_num):
    """生成Leduc Holdem训练数据"""
    raw_train_data = {
        "agent": get_agent_class(str(eval_agent)), 
        "agent_name": str(eval_agent), 
        "data": []
    }
    
    for i in range(data_num):
        while True:
            try:
                metrics = run_leduc_eval(eval_agent, LeducRandomAgent(), max_rounds=max_rounds)
                pdh = processing_leduc_decision_history(metrics['decision_history'])
                break
            except Exception as e:
                print(f"Retrying due to error: {e}")
        
        raw_train_data['data'].append(pdh)
        
        # 每100个样本输出一次调试信息
        if (i + 1) % 100 == 0:
            # 检查特征向量的差异性
            recent_vectors = raw_train_data['data'][-10:]  # 最近10个向量
            avg_vector = np.mean(recent_vectors, axis=0)
            std_vector = np.std(recent_vectors, axis=0)
            non_zero_features = np.sum(avg_vector > 0.01)
            high_variance_features = np.sum(std_vector > 0.05)
            print(f"  Sample {i+1}: Non-zero features: {non_zero_features}/60, High-variance features: {high_variance_features}/60")
    
    print(f"Generated data for {eval_agent}: {len(raw_train_data['data'])} samples")
    return raw_train_data

def generate_all_leduc_agents_data(max_rounds, data_num):
    """
    生成所有Leduc智能体的数据
    
    为了增强区分度，我们创建更极端的策略差异：
    - Random: 完全随机
    - Tight: 极度保守，只玩强牌
    - Loose: 极度松散，玩很多弱牌  
    - Aggressive: 极度激进，频繁下注加注
    - Passive: 极度被动，很少主动下注
    """
    all_agents = [
        # Random agents - 完全随机基线
        LeducRandomAgent(),
        
        # GTO agents - GTO策略
        LeducGTOAgent(player_id=0),

        # Tight agents - 极度保守策略
        LeducTightAgent(tightness=0.95, player_id=0),  # 极高保守度
        
        # Loose agents - 极度松散策略  
        LeducLooseAgent(looseness=0.95, player_id=0),  # 极高松散度
        
        # Aggressive agents - 极度激进策略
        LeducAggressiveAgent(aggression=0.95, player_id=0),  # 极高激进度
        
        # Passive agents - 极度被动策略
        LeducPassiveAgent(passiveness=0.95, player_id=0),  # 极高被动度
    ]
    
    all_data = []
    for agent in all_agents:
        print(f"Generating data for {agent}")
        data = generate_leduc_train_data(agent, max_rounds=max_rounds, data_num=data_num)
        all_data.append(data)
    
    return all_data

def train_leduc_mlp():
    """训练Leduc Holdem MLP分类器"""
    # 解析项目路径
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "leduc_holdem_mlp"
    fig_dir = base_dir / "data" / "figure"
    model_dir = base_dir / "data" / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 准备Weights & Biases日志记录
    use_wandb = True
    try:
        import wandb
    except Exception as e:
        use_wandb = False
        print("wandb 未安装，跳过 wandb 日志记录（pip install wandb 可开启）")

    # 生成或加载数据
    max_rounds = 50  # Leduc Holdem的轮数
    data_num = 100  # 减少数据量进行测试
    
    # 检查是否有缓存数据
    X_file = data_dir / f"X_{max_rounds}_{data_num}.npy"
    y_file = data_dir / f"y_{max_rounds}_{data_num}.npy"
    
    if X_file.exists() and y_file.exists():
        print("Loading cached data...")
        X = np.load(X_file)
        y = np.load(y_file)
    else:
        print("Generating new data...")
        all_data = generate_all_leduc_agents_data(max_rounds, data_num)
        
        X = []
        y = []
        for data in all_data:
            X.extend(data['data'])
            y.extend([data['agent']] * len(data['data']))
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # 保存数据
        np.save(X_file, X)
        np.save(y_file, y)
        print(f"Data saved to {X_file} and {y_file}")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {sorted(set(y))}")
    
    # 训练/测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 初始化wandb
    if use_wandb:
        mlp_params = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": 32,
            "learning_rate_init": 1e-3,
            "max_iter": 300,
            "early_stopping": True,
            "n_iter_no_change": 15,
            "validation_fraction": 0.1,
            "random_state": 42,
            "verbose": False,
        }
        wandb.init(
            project="leduc-holdem-mlp",
            name="mlp_train",
            config={
                **mlp_params,
                "scaler": "StandardScaler",
                "test_size": 0.2,
                "num_samples_total": int(X.shape[0]),
                "num_features": int(X.shape[1]),
                "max_rounds": max_rounds,
                "data_num": data_num,
            },
        )

    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP分类器配置
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )

    print("Training MLP...")
    mlp.fit(X_train_scaled, y_train)

    # 记录训练曲线
    if use_wandb:
        loss_curve = getattr(mlp, "loss_curve_", [])
        val_scores = getattr(mlp, "validation_scores_", None)
        for i, loss in enumerate(loss_curve):
            log_obj = {"iteration": i + 1, "train_loss": float(loss)}
            if isinstance(val_scores, list) and i < len(val_scores):
                log_obj["val_score"] = float(val_scores[i])
            wandb.log(log_obj, step=i + 1)

    # 评估
    y_pred_train = mlp.predict(X_train_scaled)
    y_pred = mlp.predict(X_test_scaled)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)

    # 分类报告
    unique_classes = sorted(set(int(c) for c in y.tolist()))
    class_names = [name_mapping[i] for i in unique_classes]
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "train_accuracy": float(acc_train),
        "test_accuracy": float(acc_test),
        "classes": unique_classes,
        "class_names": class_names,
        "classification_report": report_dict,
        "max_rounds": max_rounds,
        "data_num": data_num,
    }

    # 保存模型
    model_path = model_dir / "mlp_leduc_holdem.pkl"
    scaler_path = model_dir / "mlp_leduc_holdem_scaler.pkl"
    metrics_path = model_dir / "mlp_leduc_holdem_metrics.json"
    joblib.dump(mlp, model_path)
    joblib.dump(scaler, scaler_path)
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存Pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("mlp", mlp),
    ])
    pipeline_path = model_dir / "mlp_leduc_holdem_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(unique_classes)))
    ax.set_yticks(range(len(unique_classes)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (MLP on Leduc Hold\'em)')
    
    # 添加数值标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    
    fig.tight_layout()
    cm_fig_path = fig_dir / "mlp_leduc_confusion_matrix.png"
    fig.savefig(cm_fig_path)
    
    if use_wandb:
        wandb.log({"confusion_matrix": wandb.Image(str(cm_fig_path))})
    plt.close(fig)

    # 记录到wandb
    if use_wandb:
        wandb.log({
            "train_accuracy": float(acc_train),
            "test_accuracy": float(acc_test),
        })
        try:
            wandb.save(str(model_path))
            wandb.save(str(scaler_path))
            wandb.save(str(metrics_path))
            wandb.save(str(pipeline_path))
        except Exception:
            pass
        wandb.finish()

    print(f"Training done. Train acc: {acc_train:.4f}, Test acc: {acc_test:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Pipeline saved to: {pipeline_path}")
    print(f"Metrics saved to: {metrics_path}")

def t_sne_leduc_visualization(use_cached_data=True):
    """Leduc Holdem的t-SNE可视化"""
    from sklearn.manifold import TSNE
    
    max_rounds = 50
    data_num = 2000
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "leduc_holdem_mlp"
    fig_dir = base_dir / "data" / "figure"

    if not use_cached_data:
        all_data = generate_all_leduc_agents_data(max_rounds, data_num)

        X = []
        y = []
        for data in all_data:
            X.extend(data['data'])
            y.extend([data['agent']] * len(data['data']))
        
        X = np.array(X)
        y = np.array(y)

        # 保存数据
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"X_{max_rounds}_{data_num}.npy", X)
        np.save(data_dir / f"y_{max_rounds}_{data_num}.npy", y)
    else:
        X = np.load(data_dir / f"X_{max_rounds}_{data_num}.npy")
        y = np.load(data_dir / f"y_{max_rounds}_{data_num}.npy")

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Agent Types")
    plt.title("t-SNE Visualization of Leduc Hold'em Agent Strategies")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"tsne_leduc_agent_strategies_{max_rounds}_{data_num}.png")
    plt.show()

if __name__ == "__main__":
    # 可以选择运行t-SNE可视化或训练MLP
    t_sne_leduc_visualization(use_cached_data=False)
    #train_leduc_mlp()

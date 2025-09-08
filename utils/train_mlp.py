from utils.eval_llm_parallel import run_eval
from utils.eval_llm_parallel import RandomAgent, GtoAgent, BluffAgent, ValueAgent, PassiveAgent, AggressiveAgent

def get_agent_class(agent_name):
    if "RandomAgent" in agent_name:
        return 0
    elif "GtoAgent" in agent_name:
        return 1
    elif "BluffAgent" in agent_name:
        return 2
    elif "ValueAgent" in agent_name:
        return 3
    elif "PassiveAgent" in agent_name:
        return 4
    elif "AggressiveAgent" in agent_name:
        return 5
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")

name_mapping = {0:"RandomAgent", 1:"GtoAgent", 2:"BluffAgent", 3:"ValueAgent", 4:"PassiveAgent", 5:"AggressiveAgent"}

def processing_decision_history(dh):
    '''
    {"as_player_0": {"K": {"": {"[check]": 492, "[bet]": 508}, "check->bet": {"[fold]": 114, "[call]": 125}}, "Q": {"": {"[bet]": 537, "[check]": 463}, "check->bet": {"[call]": 114, "[fold]": 111}}, "J": {"": {"[bet]": 482, "[check]": 518}, "check->bet": {"[call]": 151, "[fold]": 109}}}, "as_player_1": {"K": {"check": {"[check]": 244, "[bet]": 264}, "bet": {"[call]": 240, "[fold]": 252}}, "Q": {"bet": {"[call]": 260, "[fold]": 257}, "check": {"[check]": 249, "[bet]": 234}}, "J": {"check": {"[bet]": 247, "[check]": 251}, "bet": {"[fold]": 267, "[call]": 235}}}}

    ------->
    24-dim vector: [0, 1]
    self.first_player_gto_1 = {
        "K": {'bet': 1.0, 'check': 0.0},
        "Q": {'bet': 0.0, 'check': 1.0},
        "J": {'bet': alpha, 'check': 1.0 - alpha},
    }
    self.first_player_gto_2 = {
        "K": {'call': 1.0, 'fold': 0.0},
        "Q": {'call': 0.0, 'fold': 1.0},
        "J": {'call': 0.0, 'fold': 1.0}
    }
    self.second_player_gto = {
        "K": {'bet': {'call': 1.0, 'fold': 0.0}, 'check': {'bet': 1.0, 'check': 0.0}},
        "Q": {'bet': {'call': 0.0, 'fold': 1.0}, 'check': {'bet': 0.0, 'check': 1.0}},
        "J": {'bet': {'call': 0.0, 'fold': 1.0}, 'check': {'bet': beta, 'check': 1.0 - beta}}
    }
    '''
    x = [0 for _ in range(24)]
    try:
        # as player 0
        tmp = dh['as_player_0']['K']['']
        x[0] = tmp.get('[bet]', 0) / (tmp.get('[bet]', 0) + tmp.get('[check]', 0))
        x[1] = 1 - x[0]
        if 'check->bet' not in dh['as_player_0']['K']: x[2] = x[3] = 0
        else:
            tmp = dh['as_player_0']['K']['check->bet']
            x[2] = tmp.get('[call]', 0) / (tmp.get('[call]', 0) + tmp.get('[fold]', 0))
            x[3] = 1 - x[2]
        tmp = dh['as_player_0']['Q']['']
        x[4] = tmp.get('[bet]', 0) / (tmp.get('[bet]', 0) + tmp.get('[check]', 0))
        x[5] = 1 - x[4]
        if 'check->bet' not in dh['as_player_0']['Q']: x[6] = x[7] = 0
        else:
            tmp = dh['as_player_0']['Q']['check->bet']
            x[6] = tmp.get('[call]', 0) / (tmp.get('[call]', 0) + tmp.get('[fold]', 0))
            x[7] = 1 - x[6]
        tmp = dh['as_player_0']['J']['']
        x[8] = tmp.get('[bet]', 0) / (tmp.get('[bet]', 0) + tmp.get('[check]', 0))
        x[9] = 1 - x[8]
        if 'check->bet' not in dh['as_player_0']['J']: x[10] = x[11] = 0
        else:
            tmp = dh['as_player_0']['J']['check->bet']
            x[10] = tmp.get('[call]', 0) / (tmp.get('[call]', 0) + tmp.get('[fold]', 0))
            x[11] = 1 - x[10]

        # as player 1
        tmp = dh['as_player_1']['K']['bet']
        x[12] = tmp.get('[call]', 0) / (tmp.get('[call]', 0) + tmp.get('[fold]', 0))
        x[13] = 1 - x[12]
        tmp = dh['as_player_1']['K']['check']
        x[14] = tmp.get('[bet]', 0) / (tmp.get('[bet]', 0) + tmp.get('[check]', 0))
        x[15] = 1 - x[14]
        tmp = dh['as_player_1']['Q']['bet']
        x[16] = tmp.get('[call]', 0) / (tmp.get('[call]', 0) + tmp.get('[fold]', 0))
        x[17] = 1 - x[16]
        tmp = dh['as_player_1']['Q']['check']
        x[18] = tmp.get('[bet]', 0) / (tmp.get('[bet]', 0) + tmp.get('[check]', 0))
        x[19] = 1 - x[18]
        tmp = dh['as_player_1']['J']['bet']
        x[20] = tmp.get('[call]', 0) / (tmp.get('[call]', 0) + tmp.get('[fold]', 0))
        x[21] = 1 - x[20]
        tmp = dh['as_player_1']['J']['check']
        x[22] = tmp.get('[bet]', 0) / (tmp.get('[bet]', 0) + tmp.get('[check]', 0))
        x[23] = 1 - x[22]
    except Exception as e:
        print(f"Error in processing decision history: {dh}")
        raise e
    # print(f"{dh} -> {x}")
    return x

def generate_train_data(eval_agent, each_deck_rounds, data_num):
    raw_train_data = {"agent": get_agent_class(str(eval_agent)), "agent_name": str(eval_agent), "data": []}
    for _ in range(data_num):
        while True:
            try:
                metrics = run_eval(eval_agent, RandomAgent(), each_deck_rounds=each_deck_rounds)
                pdh = processing_decision_history(metrics['decision_history'])
                break
            except Exception as e:
                print(f"Retrying due to error: {e}")

        raw_train_data['data'].append(pdh)

    print(raw_train_data)
    return raw_train_data

def generate_all_agents_data(each_deck_rounds, data_num):
    all_agents = [
        RandomAgent(),
        GtoAgent(0), GtoAgent(1/6), GtoAgent(1/3),
        BluffAgent(1/2), BluffAgent(2/3), BluffAgent(5/6), BluffAgent(1),
        BluffAgent(1/2, 1/2), BluffAgent(2/3, 2/3), BluffAgent(5/6, 5/6), BluffAgent(1, 1),
        ValueAgent(1/3), ValueAgent(1/2), ValueAgent(2/3), ValueAgent(1),
        PassiveAgent(1/3), PassiveAgent(1/2), PassiveAgent(2/3), PassiveAgent(1),
        AggressiveAgent(1/3), AggressiveAgent(1/2), AggressiveAgent(2/3), AggressiveAgent(1),
    ]
    all_data = []
    for agent in all_agents:
        print(f"Generating data for {agent}")
        data = generate_train_data(agent, each_deck_rounds=each_deck_rounds, data_num=data_num)
        all_data.append(data)
    return all_data

def t_sne_visualization(use_cached_data=True):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np

    each_deck_rounds = 10
    data_num = 1000

    if not use_cached_data:
        all_data = generate_all_agents_data(each_deck_rounds, data_num)

        X = []
        y = []
        for data in all_data:
            X.extend(data['data'])
            y.extend([data['agent']] * len(data['data']))
        
        X = np.array(X)
        y = np.array(y)

        # save X, y to npy file
        np.save(f"data/kuhn_poker_mlp/X_{each_deck_rounds}_{data_num}.npy", X)
        np.save(f"data/kuhn_poker_mlp/y_{each_deck_rounds}_{data_num}.npy", y)
    else:
        X = np.load(f"data/kuhn_poker_mlp/X_{each_deck_rounds}_{data_num}.npy")
        y = np.load(f"data/kuhn_poker_mlp/y_{each_deck_rounds}_{data_num}.npy")

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Agents")
    plt.title("t-SNE Visualization of Agent Strategies")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("data/figure/tsne_agent_strategies_10_1000.png")

def train_mlp():
    # TODO: implement MLP training
    pass

if __name__ == "__main__":
    t_sne_visualization()

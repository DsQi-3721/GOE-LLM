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

    each_deck_rounds = 50
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
        np.save(f"/home/wangxinqi/llm-opponent-modeling/data/OOD/X_{each_deck_rounds}_{data_num}.npy", X)
        np.save(f"/home/wangxinqi/llm-opponent-modeling/data/OOD/y_{each_deck_rounds}_{data_num}.npy", y)
    else:
        X = np.load(f"/home/wangxinqi/llm-opponent-modeling/data/OOD/X_{each_deck_rounds}_{data_num}.npy")
        y = np.load(f"/home/wangxinqi/llm-opponent-modeling/data/OOD/y_{each_deck_rounds}_{data_num}.npy")

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Agents")
    plt.title("t-SNE Visualization of Agent Strategies")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig("/home/wangxinqi/llm-opponent-modeling/data/figure/tsne_agent_strategies_50_1000.png")

def train_mlp():
    import json
    import os
    from pathlib import Path
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    import joblib
    import matplotlib.pyplot as plt

    # Resolve project base and important directories
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "kuhn_poker_mlp"
    fig_dir = base_dir / "data" / "figure"
    model_dir = base_dir / "data" / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prepare optional Weights & Biases logging
    use_wandb = True
    try:
        import wandb  # type: ignore
    except Exception as e:
        use_wandb = False
        print("wandb 未安装，跳过 wandb 日志记录（pip install wandb 可开启）")

    # Load all available cached datasets X_*.npy and matching y_*.npy
    X_list = []
    y_list = []
    for x_path in sorted(data_dir.glob("X_*.npy")):
        suffix = x_path.name[len("X_"):]  # e.g. "10_1000.npy"
        y_path = data_dir / f"y_{suffix}"
        if not y_path.exists():
            print(f"Skip: missing label file for {x_path.name}")
            continue
        X_chunk = np.load(x_path)
        y_chunk = np.load(y_path)
        if X_chunk.shape[0] != y_chunk.shape[0]:
            print(f"Skip: size mismatch {x_path.name} vs {y_path.name}: {X_chunk.shape[0]} != {y_chunk.shape[0]}")
            continue
        X_list.append(X_chunk)
        y_list.append(y_chunk)

    if not X_list:
        raise FileNotFoundError(f"No dataset found under {data_dir}. Expected pairs X_*.npy and y_*.npy.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Train / Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize wandb run (after数据就绪，可记录样本统计)
    if use_wandb:
        # 定义模型超参配置，便于复现
        mlp_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": 64,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "validation_fraction": 0.1,
            "random_state": 42,
            "verbose": False,
        }
        wandb.init(
            project="kuhn-poker-mlp",
            name="mlp_train",
            config={
                **mlp_params,
                "scaler": "StandardScaler",
                "test_size": 0.2,
                "num_samples_total": int(X.shape[0]),
                "num_features": int(X.shape[1]),
            },
        )


    # Feature scaling（no need）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP classifier configuration
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )

    mlp.fit(X_train_scaled, y_train)

    # Log loss/val曲线到 wandb（若可用）
    if use_wandb:
        loss_curve = getattr(mlp, "loss_curve_", [])
        val_scores = getattr(mlp, "validation_scores_", None)
        for i, loss in enumerate(loss_curve):
            log_obj = {"iteration": i + 1, "train_loss": float(loss)}
            if isinstance(val_scores, list) and i < len(val_scores):
                log_obj["val_score"] = float(val_scores[i])
            wandb.log(log_obj, step=i + 1)

    # Evaluation
    y_pred_train = mlp.predict(X_train_scaled)
    y_pred = mlp.predict(X_test_scaled)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred)

    # Classification report (test set)
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
    }

    # Save model artifacts
    model_path = model_dir / "mlp_kuhn_poker.pkl"
    scaler_path = model_dir / "mlp_kuhn_poker_scaler.pkl"
    metrics_path = model_dir / "mlp_kuhn_poker_metrics.json"
    joblib.dump(mlp, model_path)
    joblib.dump(scaler, scaler_path)
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save a single Pipeline that includes scaler + model for one-file inference
    pipeline = Pipeline([
        ("scaler", scaler),
        ("mlp", mlp),
    ])
    pipeline_path = model_dir / "mlp_kuhn_poker_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    # Confusion matrix plot (test set)
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(unique_classes)))
    ax.set_yticks(range(len(unique_classes)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (MLP on Kuhn Poker)')
    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.tight_layout()
    cm_fig_path = fig_dir / "mlp_confusion_matrix.png"
    fig.savefig(cm_fig_path)
    # Log confusion matrix image/file
    if use_wandb:
        import wandb  # type: ignore
        wandb.log({"confusion_matrix": wandb.Image(str(cm_fig_path))})
    plt.close(fig)

    # Log summary与产物到 wandb（不立即 finish，后续内联进行 OOD 评估并记录到同一 run）
    if use_wandb:
        import wandb  # type: ignore
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

    # ===== 内联 OOD 评估开始 =====
    import re
    ood_dir = base_dir / "data" / "OOD"

    # 固定类别顺序与名称，确保混淆矩阵与报告一致
    label_order = sorted(name_mapping.keys())
    label_names = [name_mapping[i] for i in label_order]

    per_dataset_metrics = {}
    all_X = []
    all_y = []
    per_points = []  # (edr, dnum, acc, n)

    for x_path in sorted(ood_dir.glob("X_*.npy")):
        suffix = x_path.name[len("X_"):]
        y_path = ood_dir / f"y_{suffix}"
        if not y_path.exists():
            print(f"Skip OOD: missing label file for {x_path.name}")
            continue

        Xo = np.load(x_path).astype(np.float32)
        yo = np.load(y_path).astype(np.int64)
        if Xo.shape[0] != yo.shape[0]:
            print(f"Skip OOD: size mismatch {x_path.name} vs {y_path.name}: {Xo.shape[0]} != {yo.shape[0]}")
            continue

        Xo_scaled = scaler.transform(Xo)
        yo_pred = mlp.predict(Xo_scaled)

        acc_o = accuracy_score(yo, yo_pred)
        cm_o = confusion_matrix(yo, yo_pred, labels=label_order)

        # 绘制并保存 OOD 混淆矩阵
        fig_o, ax_o = plt.subplots(figsize=(8, 6))
        im = ax_o.imshow(cm_o, cmap='Blues')
        ax_o.figure.colorbar(im, ax=ax_o)
        ax_o.set_xticks(range(len(label_order)))
        ax_o.set_yticks(range(len(label_order)))
        ax_o.set_xticklabels(label_names, rotation=45, ha='right')
        ax_o.set_yticklabels(label_names)
        ax_o.set_xlabel('Predicted label')
        ax_o.set_ylabel('True label')
        title_suffix = suffix.replace('.npy', '')
        ax_o.set_title(f'Confusion Matrix OOD ({title_suffix})')
        for i in range(cm_o.shape[0]):
            for j in range(cm_o.shape[1]):
                ax_o.text(j, i, int(cm_o[i, j]), ha='center', va='center', color='black')
        fig_o.tight_layout()
        cm_o_path = fig_dir / f"mlp_confusion_matrix_ood_{title_suffix}.png"
        fig_o.savefig(cm_o_path)
        plt.close(fig_o)

        per_dataset_metrics[title_suffix] = {
            "num_samples": int(Xo.shape[0]),
            "accuracy": float(acc_o),
            "confusion_matrix_png": str(cm_o_path),
        }

        all_X.append(Xo)
        all_y.append(yo)

        # 解析 each_deck_rounds 与 data_num
        m = re.match(r"(\d+)_([0-9]+)$", title_suffix)
        if m:
            edr = int(m.group(1))
            dnum = int(m.group(2))
        else:
            parts = title_suffix.split('_')
            edr = int(parts[0]) if parts and parts[0].isdigit() else -1
            dnum = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        per_points.append((edr, dnum, float(acc_o), int(Xo.shape[0])))

        if use_wandb:
            wandb.log({
                "ood/each_deck_rounds": edr,
                "ood/data_num": dnum,
                "ood/num_samples": int(Xo.shape[0]),
                "ood/accuracy": float(acc_o),
            })
            wandb.log({f"ood/conf_mat_{title_suffix}": wandb.Image(str(cm_o_path))})

    # 总体 OOD 评估与曲线
    if per_dataset_metrics:
        X_all = np.vstack(all_X).astype(np.float32)
        y_all = np.concatenate(all_y).astype(np.int64)
        X_all_scaled = scaler.transform(X_all)
        y_all_pred = mlp.predict(X_all_scaled)
        acc_all = accuracy_score(y_all, y_all_pred)
        cm_all = confusion_matrix(y_all, y_all_pred, labels=label_order)

        fig_all, ax_all = plt.subplots(figsize=(8, 6))
        im = ax_all.imshow(cm_all, cmap='Blues')
        ax_all.figure.colorbar(im, ax=ax_all)
        ax_all.set_xticks(range(len(label_order)))
        ax_all.set_yticks(range(len(label_order)))
        ax_all.set_xticklabels(label_names, rotation=45, ha='right')
        ax_all.set_yticklabels(label_names)
        ax_all.set_xlabel('Predicted label')
        ax_all.set_ylabel('True label')
        ax_all.set_title('Confusion Matrix OOD (overall)')
        for i in range(cm_all.shape[0]):
            for j in range(cm_all.shape[1]):
                ax_all.text(j, i, int(cm_all[i, j]), ha='center', va='center', color='black')
        fig_all.tight_layout()
        cm_all_path = fig_dir / "mlp_confusion_matrix_ood_overall.png"
        fig_all.savefig(cm_all_path)
        plt.close(fig_all)

        # 折线图：each_deck_rounds vs accuracy
        per_points_sorted = sorted(per_points, key=lambda t: t[0])
        xs = [p[0] for p in per_points_sorted if p[0] >= 0]
        ys = [p[2] for p in per_points_sorted if p[0] >= 0]
        if xs and ys:
            fig_curve, ax_curve = plt.subplots(figsize=(8, 4))
            ax_curve.plot(xs, ys, marker='o')
            ax_curve.set_xlabel('each_deck_rounds (OOD)')
            ax_curve.set_ylabel('Accuracy')
            ax_curve.set_title('OOD Accuracy vs each_deck_rounds')
            ax_curve.grid(True, linestyle='--', alpha=0.4)
            curve_path = fig_dir / 'mlp_ood_accuracy_curve.png'
            fig_curve.tight_layout()
            fig_curve.savefig(curve_path)
            plt.close(fig_curve)
        else:
            curve_path = None

        # 保存 OOD 汇总 JSON
        ood_results = {
            "overall": {
                "num_samples": int(X_all.shape[0]),
                "accuracy": float(acc_all),
                "confusion_matrix_png": str(cm_all_path),
            },
            "per_dataset": per_dataset_metrics,
        }
        ood_json_path = model_dir / "mlp_kuhn_poker_metrics_ood.json"
        with ood_json_path.open('w', encoding='utf-8') as f:
            json.dump(ood_results, f, ensure_ascii=False, indent=2)

        print(f"[Train] OOD evaluation done. Overall OOD acc: {acc_all:.4f}")

        if use_wandb:
            wandb.log({
                "ood/overall_accuracy": float(acc_all),
                "ood/conf_mat_overall": wandb.Image(str(cm_all_path)),
            })
            if curve_path is not None:
                wandb.log({"ood/accuracy_curve": wandb.Image(str(curve_path))})

    # 结束同一 wandb 运行
    if use_wandb:
        wandb.finish()
    # ===== 内联 OOD 评估结束 =====

    print(f"Training done. Train acc: {acc_train:.4f}, Test acc: {acc_test:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Pipeline (scaler+model) saved to: {pipeline_path}")
    print(f"Metrics saved to: {metrics_path}")

def eval_mlp():
    raise NotImplementedError("eval_mlp 已合并进 train_mlp() 内联流程，无需单独调用。")

if __name__ == "__main__":
    #t_sne_visualization(use_cached_data=False)
    train_mlp()

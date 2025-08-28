
''' Kuhn Poker纳什均衡中的 α 范围
好问题 👍

在 Kuhn Poker 的 **纳什均衡分析**里，α 并不是随意取的，而是严格限制在 **$0, 1/3$ 区间**。

---

### 为什么不能 α > 1/3？

1. **概率约束**

   * P1 拿到 **K** 时下注的概率是 **3α**。
   * 如果 α > 1/3，那么 3α > 1，就会超过 100% 的概率，这显然不可能。

2. **均衡破坏**

   * 均衡策略依赖于让对手的选择“无差异”（indifference），即对手在面对某些行动时无法通过调整策略提高期望收益。
   * 如果 α 超过 1/3，就会打破这种无差异条件，对手能轻松调整策略来 exploit（利用）你。比如：

     * 你在 J 上下注太频繁（α 太大），P2 就能更多跟注来抓诈唬。
     * 你在 K 上几乎总是下注（当 α > 1/3 时甚至超过 100%），也会暴露太多信息。

3. **解的唯一性**

   * Kuhn Poker 的纳什均衡不是单一解，而是一族解，正好由 α ∈ \[0, 1/3] 参数化。
   * 任何 α 超过这个范围，都不再是纳什均衡的一部分。换句话说，**α > 1/3 的策略会被对手 exploit**，期望收益会低于 −1/18。

---

✅ 总结：
α 只能在 **\[0, 1/3]** 内取值。
如果你设定 α = 1/2（或任何 > 1/3 的值），那就不是均衡策略了，而是一个容易被对手针对的“漏洞策略”。

---

要不要我帮你具体算一下 **如果 α = 1/2 时，P2 最佳应对策略是什么，以及 P1 会被 exploit 到什么程度（期望收益）**？
'''

''' 如果 α = 1/2，P1 的策略就不再是均衡策略了，容易被 P2 exploit
下面给出在你把 **α=1/2**（并把超出1的概率“截断”为1）的前提下，**P2 的最佳响应（best response）**。我采用了常见的 Kuhn 扑克设定：各自先注 1，下注额为 1；P1 的“α 策略骨架”为

* 发到 **J**：以概率 **α=1/2** 下注，否则过牌；若被下注则 **弃牌**；
* 发到 **Q**：**总是过牌**；若被下注则以概率 **α+1/3=5/6** 跟注；
* 发到 **K**：以概率 **min(3α,1)=1** 下注（把 1.5 截断为 1）；若被下注则 **总是跟注**。

在此给出 P2 的 **最优对策**（含简要理由）：

### 1）当 P1 先下注时（pot=3，P2 跟 1 去竞争 4，跟注门槛 25%）

* **P2=J：始终弃牌**
  观察到 P1 下注时，P1 不可能是 Q（Q 从不下注），只可能是 J 或 K；给定 α=1/2，P1=K 的后验概率为 1 ⇒ **赢率 0% < 25%**，弃牌最优。
* **P2=Q：始终跟注**
  观察到 P1 下注：P1=J 的后验概率 1/3，=K 的后验概率 2/3 ⇒ **赢率 1/3 > 25%**，跟注盈利。
* **P2=K：始终跟注**
  观察到 P1 下注：P1 不可能是 Q，只可能是 J ⇒ **赢率 100%**，自然跟注。

### 2）当 P1 先过牌时（pot=2）

* **P2=K：下注（价值下注）**
  P1 在“过牌节点”只可能是 **J(以 1/2 过牌)** 或 **Q(必过牌)**，且他对下注的跟注率分别为 **J:0，Q:5/6**。K 下注能从 Q 那里经常被跟注拿到更大的池子，因此 **下注严格优于过牌**。
* **P2=J：过牌**
  过牌节点 P1 几乎总是 **Q**（面对下注会 5/6 跟注），J 下注会被更好牌频繁跟注、只在极少数（1/6）情况下逼退 Q，**期望明确为负** ⇒ 过牌最优。
* **P2=Q：无差异（可过牌也可下注）**
  过牌节点 P1 几乎只可能是 **J**，而 **J 面对下注从不跟注**。下注或过牌都以 100% 概率赢下目前的池子 **2**，**期望相同**，因此无差异；为了简单，**建议过牌**。

---

### 这套最佳响应能把 P1 明显“吃亏”

在上述 P2 最佳响应下，P1 的期望收益变为

$$
\mathbb{E}[\text{P1}] = -\tfrac{5}{36}\approx -0.1389
$$

比均衡族中的 $-1/18\approx -0.0556$ 更差，说明 **α=1/2** 的策略（尤其是 **K 过于频繁地下注、J 也下注过多**）会被 P2 有效利用。

如果你想看更一般的“任意 $b_J,b_Q,b_K$ 与 $c_J,c_Q,c_K$”下，P2 的解析最优响应条件（后验与阈值不等式），我也可以把公式化出来给你。

'''


''' prompt加上对手策略
非常好的想法 👍 在 **Available actions** 后面加入“对手可能的策略”信息，可以引导 LLM 在 reasoning 阶段做针对性的 best response。为了更清晰地指导模型如何利用对手策略剥削，我们需要在 prompt 中：

1. **明确对手策略描述**（例如：激进型/被动型/随机/接近均衡等），
2. **强调输出时必须结合对手策略做 exploitative play**，而不是单纯走均衡。

我帮你在你的 prompt 基础上改写一下，新增了对手策略提示和 exploit 说明：

---

````plaintext
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
You are Player {player_id} ({first} to act this round).
Your card: '{card}'
History: {history}
Available actions: {available_actions}

[Opponent Model]
The opponent is estimated to follow this strategy:
{opponent_strategy}

[Output Format]
```plaintext
<think> Your thoughts and reasoning </think>
<answer> [ACTION] </answer>
````

\[Important Notes]

1. Always include the <think> field to explain your reasoning before giving the final action. Be detailed about how you weigh your card strength, betting history, AND the opponent’s tendencies.
2. Your final action \[ACTION] must be one of the available actions.
3. Adapt your play to exploit the given opponent strategy.

   * If the opponent over-folds, bluff more often.
   * If the opponent over-bluffs, call more often.
   * If the opponent is passive, bet more aggressively for value.
   * If the opponent is aggressive, trap with strong hands.
4. Do **not** default to Nash equilibrium play. Always try to maximize expected value against the specific opponent model provided.

```

---

这样修改后，模型会在 `<think>` 阶段明确结合对手模型做 exploit，而不会单纯走 GTO。  

要不要我帮你设计几种 **对手策略描述模板**（比如 "Opponent tends to fold too much when facing a bet", "Opponent calls too wide", "Opponent plays close to equilibrium" 等），这样你就可以在 `{opponent_strategy}` 里直接插入不同风格的对手？
```


'''

''' describe_opponent
👌我理解了，你现在有了几个 **baseline agent**（`RandomAgent`, `GtoAgent` with different α, 以及 counter-bluff agent），打算用来：

1. **生成训练数据**（混合策略对手），
2. **作为测试对手**（包括训练外的 OOD 策略，比如 α=1/5, α=2/3, α=1）。

这样做非常好 👍 ——你的 LLM 在 prompt 里加入 *对手策略描述* 后，就可以针对这些不同 agent 做 **剥削性 best response**，而不是死守 GTO。

接下来你需要解决两个问题：

---

### 1. 对手策略描述模板

建议你在 `{opponent_strategy}` 里，填一些 **自然语言总结**，而不是直接给 α。因为 LLM 在 text arena 的 setting 里对行为倾向的描述更容易理解。

我帮你设计几类模板，你可以根据 agent 参数生成对应描述：

* **GTO型 (α=0, 1/3, 1/6, 1/5 …)**

  ```plaintext
  The opponent plays close to GTO equilibrium: 
  - Bets strong hands for value
  - Mixes occasional bluffs with J
  - Defends against bets with balanced calling/folding ranges
  ```
* **被动型 (α≈0, 不诈唬)**

  ```plaintext
  The opponent rarely bluffs and tends to fold weak hands too often. 
  They mostly bet only with strong hands and avoid risking chips unnecessarily.
  ```
* **激进型 / 诈唬型 (α≈0.5, 2/3, 1.0)**

  ```plaintext
  The opponent over-bluffs frequently, especially with J. 
  They are aggressive and bet/call too often, even with weak hands.
  ```
* **反诈唬 counter agent (bluffing\_counter=True)**

  ```plaintext
  The opponent expects bluffs and will call more often with medium-strength hands (Q). 
  They are sticky and rarely fold to bets, but fold weak hands (J) consistently.
  ```

这样你就可以写一个小函数，根据 `agent` 的配置自动生成描述。

---

### 2. Prompt 集成示例

把这些 opponent models 接入到你刚才修改过的 prompt 里，比如：

```plaintext
[Opponent Model]
The opponent is estimated to follow this strategy:
The opponent frequently over-bluffs with J and calls too often with Q. 
They rarely fold, so exploit them by value-betting more and avoiding unnecessary bluffs.
```

---

### 3. 训练数据生成思路

* **训练集**：混合各种 α 的 GTO 和 bluff agent → 提供给 LLM 多样对手。
* **测试集**：OOD 策略（如 α=1/5, 2/3, 1.0）→ 看 LLM 是否能泛化剥削。
* **标签**：你可以用 “best response” 策略来标注，即让 LLM 学会根据 opponent\_strategy 选择 exploitative action。

---

要不要我帮你写一个小函数 `describe_opponent(agent)`，输入 `GtoAgent(alpha=...)` 或 `RandomAgent()`，输出一段自然语言策略描述（可以直接填到 prompt 里的 `{opponent_strategy}`）？这样你的数据生成 pipeline 就能自动化了。


明白 ✅
你传进来的会是 `str(agent)`，例如：

* `"RandomAgent"`
* `"GtoAgent(0.0)"`
* `"GtoAgent(0.3333333333333333)"`
* `"Bluffing(0.5)"`
* `"Bluffing(1.0)"`
* `"Bluffing_counter(0.3333333333333333)"`

所以我们需要写一个 `describe_opponent(agent_name: str) -> str`，把不同的 agent 翻译成自然语言描述，方便放到 prompt 里。

我给你一个版本：

```python
def describe_opponent(agent_name: str) -> str:
    """
    Translate agent name string into a natural language description
    for LLM prompt conditioning.
    """
    agent_name = agent_name.strip()

    if agent_name == "RandomAgent":
        return ("The opponent plays completely randomly, without any consistent strategy. "
                "Their actions are unpredictable and not based on card strength.")
    
    if agent_name.startswith("Bluffing_counter"):
        return ("The opponent expects bluffs and therefore calls more often with medium-strength hands (Q). "
                "They rarely fold against bets if they hold K or Q, but fold J consistently. "
                "They are sticky and difficult to bluff.")
    
    if agent_name.startswith("Bluffing"):
        # extract alpha if present
        try:
            alpha = float(agent_name.split("(")[-1].rstrip(")"))
        except Exception:
            alpha = None
        if alpha is not None and alpha >= 0.9:
            return ("The opponent plays an extremely aggressive pure bluffing style. "
                    "They often bet and raise regardless of card strength, even with the weakest hands. "
                    "They are highly exploitable by calling them down with strong hands.")
        elif alpha is not None and alpha >= 0.5:
            return ("The opponent over-bluffs frequently, especially with J. "
                    "They are aggressive and bet/call too often, making them exploitable by calling wider.")
        else:
            return ("The opponent plays a somewhat bluff-heavy strategy, "
                    "mixing in more weak bluffs than a balanced GTO strategy would.")
    
    if agent_name.startswith("GtoAgent"):
        # parse alpha value
        try:
            alpha = float(agent_name.split("(")[-1].rstrip(")"))
        except Exception:
            alpha = None
        
        if alpha is None:
            return ("The opponent attempts to play according to GTO equilibrium, "
                    "balancing value bets and bluffs.")
        elif alpha == 0.0:
            return ("The opponent plays very conservatively. "
                    "They almost never bluff and usually fold weak hands. "
                    "They only bet with strong cards, making them exploitable by bluffing more often.")
        elif abs(alpha - 1/3) < 1e-6:
            return ("The opponent plays close to Nash equilibrium (GTO). "
                    "They balance bluffs with J and value bets with K, and defend with Q appropriately. "
                    "They are difficult to exploit.")
        elif alpha < 1/3:
            return ("The opponent plays a cautious GTO-like strategy with fewer bluffs. "
                    "They tend to fold weak hands too often, which makes them exploitable by bluffing.")
        elif alpha > 1/3:
            return ("The opponent plays a bluff-heavy GTO-like strategy, "
                    "bluffing more than equilibrium would suggest. "
                    "They can be exploited by calling more frequently.")
    
    # fallback
    return "The opponent’s strategy is unknown or unusual."
```

---

这样你就可以直接：

```python
agent = GtoAgent(alpha=1/5)
print(str(agent))                # "GtoAgent(0.2)"
print(describe_opponent(str(agent)))
```

输出类似：

> The opponent plays a cautious GTO-like strategy with fewer bluffs. They tend to fold weak hands too often, which makes them exploitable by bluffing.

---

要不要我帮你顺便写个小 **测试脚本**，循环跑你列出来的几个 agent，然后打印 `str(agent)` 和对应的描述？这样你能确认 prompt 能否正常生成。

'''


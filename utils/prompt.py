prompt_template_old = """
[GAME] You are Player 0 in Kuhn Poker.
Game Rules:
- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)
- Each player antes 1 chip and receives 1 card each round (note that the cards are dealt without replacement, so you cannot have the same card as your opponent).
- The player with the highest card wins the pot

Action Rules:
- '[check]': Pass without betting (only if no bet is on the table)
- '[bet]': Add 1 chip to the pot (only if no bet is on the table)
- '[call]': Match an opponent's bet by adding 1 chip to the pot
- '[fold]': Surrender your hand and let your opponent win the pot
- Note: You must respond with one of the actions in square brackets, '[ACTION]'.

[GAME] Your card is: 'K'
[GAME] Your available actions are: '[check]', '[bet]'
[GAME] Player 0, submitted move: '[check]'.
[GAME] Player 1, submitted move: '[bet]'.
[GAME] Your available actions are: '[fold]', '[call]'
"""

prompt_template = """You are an expert Kuhn Poker player.

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

[Output Format]
``` plaintext
<think> Your thoughts and reasoning </think>
<answer> [ACTION] </answer>
```

[Important Notes]
1. You must always include the <think> field to outline your reasoning.
2. Your final action [ACTION] must be one of the available actions.
"""

# opponent ranges, betting patterns, and card strengths
# You may reason about the opponent's ranges, betting patterns, and card strengths.

prompt_template_opponent = """You are an expert Kuhn Poker player.

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
The opponent is estimated to follow this strategy: {opponent_description}.
You may reason about the opponent's ranges, betting patterns, and card strengths.

[Output Format]
``` plaintext
<think> Your thoughts and reasoning </think>
<answer> [ACTION] </answer>
```

[Important Notes]
1. You must always include the <think> field to outline your reasoning.
2. Your final action [ACTION] must be one of the available actions.
"""

prompt_template_opponent_analysis = """You are an expert Kuhn Poker strategist. Your goal is to analyze the opponent’s playing style and strategy based on the given history of past games.

[Game Rules]
- Kuhn Poker uses a 3-card deck: J (lowest), Q, K (highest).
- Each player antes 1 chip and gets 1 private card.
- Action space: [check], [bet], [call], [fold].

[Past Rounds]
{opponent_histories}

[Output Format]
```plaintext
<think> Your thoughts and reasoning </think>
<summary> Concise natural language description of the opponent’s strategy. </summary>
```

[Analysis Guidelines]
- Estimate how often the opponent bets, checks, calls, and folds in different situations (first to act, second to act, after facing a bet, with strong vs weak cards if revealed).  
- Highlight any consistent patterns, such as “bets aggressively when first to act,” or “rarely bluffs.”  
- If data is limited, provide cautious estimates and note uncertainty.  
- Your output will be used as input for a decision-making model in the next stage, so make it **concise, structured, and practical**.
"""

prompt_template_leduc = """You are an expert Leduc Hold'em player.

[Game Rules]
- Leduc Hold'em uses a 6-card deck: JJ, QQ, KK (two of each rank).
- Each player antes 1 chip and receives 1 private card.
- Pre-flop betting round: fixed bet size is 2 chips, maximum 2 raises per round.
- One public card is revealed, then second betting round: fixed bet size is 4 chips.
- Showdown: pair beats high card; ties split the pot.

[Action Rules]
- [check]: Pass without betting (only available when no bet is on the table)
- [bet]: Add chips to the pot (2 chips pre-flop, 4 chips post-flop; only available when no bet is on the table)
- [raise]: Increase the current bet by adding more chips (2 chips pre-flop, 4 chips post-flop; only available when there's a bet and fewer than 2 raises this round)
- [call]: Match the current bet amount (only available when there's a bet on the table)
- [fold]: Surrender your hand and let your opponent win the pot (only available when there's a bet on the table)

[State]
You are Player {player_id} ({first} to act this round).
Your card: '{card}'
{board_card}History: {history}
Available actions: {available_actions}

[Output Format]
``` plaintext
<think> Your thoughts and reasoning </think>
<answer> [ACTION] </answer>
```

[Important Notes]
1. You must always include the <think> field to outline your reasoning.
2. Your final action [ACTION] must be one of the available actions.
"""

prompt_template_leduc_opponent = """You are an expert Leduc Hold'em player.

[Game Rules]
- Leduc Hold'em uses a 6-card deck: JJ, QQ, KK (two of each rank).
- Each player antes 1 chip and receives 1 private card.
- Pre-flop betting round: fixed bet size is 2 chips, maximum 2 raises per round.
- One public card is revealed, then second betting round: fixed bet size is 4 chips.
- Showdown: pair beats high card; ties split the pot.

[Action Rules]
- [check]: Pass without betting (only available when no bet is on the table)
- [bet]: Add chips to the pot (2 chips pre-flop, 4 chips post-flop; only available when no bet is on the table)
- [raise]: Increase the current bet by adding more chips (2 chips pre-flop, 4 chips post-flop; only available when there's a bet and fewer than 2 raises this round)
- [call]: Match the current bet amount (only available when there's a bet on the table)
- [fold]: Surrender your hand and let your opponent win the pot (only available when there's a bet on the table)

[State]
You are Player {player_id} ({first} to act this round).
Your card: '{card}'
{board_card}History: {history}
Available actions: {available_actions}

[Opponent Model]
The opponent is estimated to follow this strategy: {opponent_description}.
You may reason about the opponent's ranges, betting patterns, and card strengths.

[Output Format]
``` plaintext
<think> Your thoughts and reasoning </think>
<answer> [ACTION] </answer>
```

[Important Notes]
1. You must always include the <think> field to outline your reasoning.
2. Your final action [ACTION] must be one of the available actions.
"""

prompt_template_leduc_opponent_analysis = """You are an expert Leduc Hold'em strategist. Your goal is to analyze the opponent's playing style and strategy based on the given history of past games.

[Game Rules]
- Leduc Hold'em uses a 6-card deck: JJ, QQ, KK (two of each rank).
- Each player antes 1 chip and receives 1 private card.
- Pre-flop betting round: fixed bet size is 2 chips, maximum 2 raises per round.
- One public card is revealed, then second betting round: fixed bet size is 4 chips.
- Action space: [check], [bet], [raise], [call], [fold].

[Past Rounds]
{opponent_histories}

[Output Format]
```plaintext
<think> Your thoughts and reasoning </think>
<summary> Concise natural language description of the opponent's strategy. </summary>
```

[Analysis Guidelines]
- Estimate how often the opponent bets, raises, checks, calls, and folds in different situations (pre-flop vs post-flop, first to act vs second to act, after facing a bet/raise, with strong vs weak cards if revealed).  
- Highlight any consistent patterns, such as "bets aggressively pre-flop," "rarely bluffs post-flop," or "tends to fold to raises."  
- Consider the opponent's behavior in different betting rounds and with different board cards.
- If data is limited, provide cautious estimates and note uncertainty.  
- Your output will be used as input for a decision-making model in the next stage, so make it **concise, structured, and practical**.
"""


# Reference prompt from ToolRL
'''{'content': 'You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.\n\n**Available Tools**\nIn your response, you can use the following tools:\n1. Name: checkMembership\nDescription: Check if a person is a member of the library and has access to the library services\nParameters: {"user_id": {"description": "The unique identifier of the library user (e.g., library card number, username)", "type": "string", "default": ""}, "pin": {"description": "The personal identification number of the library user", "type": "string", "default": ""}}\n2. Name: logAccessEvent\nDescription: Log an access event within the library for auditing and security purposes\nParameters: {"user_id": {"description": "The unique identifier of the library user (e.g., library card number, username)", "type": "string", "default": ""}, "event_type": {"description": "The type of access event (e.g., entry, exit, resource access)", "type": "string", "default": ""}}\n3. Name: authorizeEntry\nDescription: Authorize entry of a person into the library premises\nParameters: {"user_id": {"description": "The unique identifier of the library user (e.g., library card number, username)", "type": "string", "default": ""}, "pin": {"description": "The personal identification number of the library user", "type": "string", "default": ""}}\n4. Name: getLibraryAccessControl\nDescription: Check the access control settings in a library\nParameters: {"library_name": {"description": "The name of the library you want to check the access control", "type": "string", "default": ""}, "user_id": {"description": "The ID of the user requesting access control information", "type": "string", "default": ""}, "time_of_day": {"description": "Specify a time of day for access control (e.g., morning, afternoon, evening)", "type": "string", "default": ""}}\n\n**Steps for Each Turn**\n1. **Think:** Recall relevant context and analyze the current user goal.\n2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.\n3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.\n\n**Output Format**\n```plaintext\n<think> Your thoughts and reasoning </think>\n<tool_call>\n{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}\n{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}\n...\n</tool_call>\n<response> AI\'s final response </response>\n```\n\n**Important Notes**\n1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.\n2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.\n3. Refer to the previous dialogue records in the history, including the user\'s queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).', 'role': 'system'}'''
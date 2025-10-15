from typing import Dict, Any

def create_board_str(game_state: Dict[str, Any]) -> str:
    def rank_to_str(rank: int) -> str: return {0: 'J', 1: 'Q', 2: 'K'}.get(rank, '?')
    
    pot = game_state.get("pot", 0)
    player_bank = game_state.get("player_bank", {0: 100, 1: 100})
    hands_dealt = game_state.get("hands_dealt", 0)
    round_num = game_state.get("round", 0)
    cards = game_state.get("player_cards", {0: None, 1: None})
    board_card = game_state.get("board_card", None)
    current_bet = game_state.get("current_bet", 0)
    
    p0_card = rank_to_str(cards.get(0)) if cards.get(0) is not None else "?"
    p1_card = rank_to_str(cards.get(1)) if cards.get(1) is not None else "?"
    board_card_str = rank_to_str(board_card) if board_card is not None else "?"
    
    p0_bank = f"{player_bank.get(0, 0):>3}"
    p1_bank = f"{player_bank.get(1, 0):>3}"
    
    round_info = f"Hand: {hands_dealt:<2} Round: {round_num}"
    pot_info = f"Pot: {pot:<3}"
    bet_info = f"Bet: {current_bet:<3}" if current_bet > 0 else "Bet: 0  "
    
    board_str = f"""
┌─────────────────────────────┐
│ {round_info} │
│ {pot_info} {bet_info} │
└─────────────────────────────┘

┌── P0 ({p0_bank}$) ──┐    ┌── P1 ({p1_bank}$) ──┐
│               │    │               │
│  ┌─────────┐  │    │  ┌─────────┐  │
│  │ {p0_card}       │  │    │  │ {p1_card}       │  │
│  │         │  │    │  │         │  │
│  │    ♥    │  │    │  │    ♠    │  │
│  │         │  │    │  │         │  │
│  │       {p0_card} │  │    │  │       {p1_card} │  │
│  └─────────┘  │    │  └─────────┘  │
│               │    │               │
└───────────────┘    └───────────────┘

┌─────── Board ───────┐
│                     │
│  ┌───────────────┐  │
│  │ {board_card_str}             │  │
│  │               │  │
│  │      ♦        │  │
│  │               │  │
│  │             {board_card_str} │  │
│  └───────────────┘  │
│                     │
└─────────────────────┘
""".strip()
    return board_str

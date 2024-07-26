import random
import numpy as np

def player(prev_play, opponent_history=[]):
    n_states = 9  # 3 (возможные ходы противника) * 3 (возможные ходы игрока)
    n_actions = 3  # Камень, ножницы, бумага
    alpha = 0.5  # коэффициент обучения
    gamma = 0.2  # коэффициент дисконтирования
    epsilon = 0.3  # вероятность случайного выбора действия

    actions = ["R", "P", "S"]

    if not hasattr(player, "Q"):
        player.Q = np.zeros((n_states, n_actions))
        player.last_opponent_move = None
        player.last_player_move = None
        player.last_action = None

    if prev_play:
        if player.last_opponent_move is not None and player.last_player_move is not None:
            last_state = get_state(player.last_opponent_move, player.last_player_move)
            last_action = player.last_action
            reward = get_reward(prev_play, actions[last_action])
            current_state = get_state(prev_play, actions[last_action])
            player.Q[last_state, last_action] = (1 - alpha) * player.Q[last_state, last_action] + alpha * (reward + gamma * np.max(player.Q[current_state]))

        # Обновляем последние ходы
        player.last_opponent_move = prev_play
        player.last_player_move = actions[player.last_action] if player.last_action is not None else None

    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(n_actions))
    else:
        current_state = get_state(player.last_opponent_move, player.last_player_move) if player.last_opponent_move and player.last_player_move else 0
        action = np.argmax(player.Q[current_state])

    player.last_action = action
    return actions[action]

def get_state(opponent_move, player_move):
    if opponent_move is None or player_move is None:
        return 0
    move_map = {"R": 0, "P": 1, "S": 2}
    return move_map[opponent_move] * 3 + move_map[player_move]

def get_reward(opponent_move, player_move):
    if opponent_move == player_move:
        return 0
    elif (opponent_move == "R" and player_move == "P") or (opponent_move == "P" and player_move == "S") or (opponent_move == "S" and player_move == "R"):
        return 1
    else:
        return -1

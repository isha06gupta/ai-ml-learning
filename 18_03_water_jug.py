import math

A = 4
B = 3
T = 2


def get_moves(state):
    x, y = state
    moves = []

    if x < A:
        moves.append((A, y))
    if y < B:
        moves.append((x, B))

    if x > 0:
        moves.append((0, y))
    if y > 0:
        moves.append((x, 0))

    if x > 0 and y < B:
        pour = min(x, B - y)
        moves.append((x - pour, y + pour))

    if y > 0 and x < A:
        pour = min(y, A - x)
        moves.append((x + pour, y - pour))

    # remove same state
    moves = [m for m in set(moves) if m != state]

    return moves


def is_goal(state):
    return state[0] == T or state[1] == T


# simple heuristic
def evaluate(state):
    x, y = state

    if is_goal(state):
        return 100

    dist = min(abs(x - T), abs(y - T))
    return -dist


def alpha_beta(state, depth, alpha, beta, maximizing, visited):
    if depth == 0 or is_goal(state):
        return evaluate(state)

    visited.add(state)

    if maximizing:
        max_eval = -math.inf
        for move in get_moves(state):

            # ❗ BLOCK RESET TO (0,0)
            if move == (0, 0):
                continue

            if move in visited:
                continue

            val = alpha_beta(move, depth - 1, alpha, beta, False, visited)
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)

            if beta <= alpha:
                break

        visited.remove(state)
        return max_eval

    else:
        min_eval = math.inf
        for move in get_moves(state):

            if move == (0, 0):
                continue

            if move in visited:
                continue

            val = alpha_beta(move, depth - 1, alpha, beta, True, visited)
            min_eval = min(min_eval, val)
            beta = min(beta, val)

            if beta <= alpha:
                break

        visited.remove(state)
        return min_eval


def best_move(state):
    best_val = -math.inf
    best_state = None

    for move in get_moves(state):

        # ❗ BLOCK RESET HERE ALSO
        if move == (0, 0):
            continue

        val = alpha_beta(move, 6, -math.inf, math.inf, False, set())

        if val > best_val:
            best_val = val
            best_state = move

    return best_state


def play_game():
    state = (0, 0)
    print("Game Start! Target:", T)

    while True:
        print("\nCurrent State:", state)

        moves = get_moves(state)
        print("\nYour moves:")
        for i, m in enumerate(moves):
            print(f"{i}: {m}")

        try:
            choice = int(input("Choose move: "))
            if choice < 0 or choice >= len(moves):
                print("Invalid choice.")
                continue
        except:
            print("Enter valid number.")
            continue

        state = moves[choice]

        if is_goal(state):
            print("You win!")
            break

        print("\nAI is thinking...")
        ai = best_move(state)

        if ai is None:
            print("Draw!")
            break

        state = ai
        print("AI chose:", state)

        if is_goal(state):
            print("AI wins!")
            break


if __name__ == "__main__":
    play_game()

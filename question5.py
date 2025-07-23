def evaluate(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2]:
            if board[i][0] == 'X':
                return 1
            elif board[i][0] == 'O':
                return -1
        if board[0][i] == board[1][i] == board[2][i]:
            if board[0][i] == 'X':
                return 1
            elif board[0][i] == 'O':
                return -1

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == 'X':
            return 1
        elif board[0][0] == 'O':
            return -1
    if board[0][2] == board[1][1] == board[2][0]:
        if board[0][2] == 'X':
            return 1
        elif board[0][2] == 'O':
            return -1

    # Check for draw or game not finished
    for row in board:
        for cell in row:
            if cell not in ['X', 'O']:
                return None  # Game not finished

    return 0  # Draw

nodes_examined = 0

def minimax(board, is_maximizing):
    global nodes_examined
    nodes_examined += 1

    result = evaluate(board)
    if result is not None:
        return result

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] not in ['X', 'O']:
                    board[i][j] = 'X'
                    score = minimax(board, False)
                    board[i][j] = str(3 * i + j + 1)
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] not in ['X', 'O']:
                    board[i][j] = 'O'
                    score = minimax(board, True)
                    board[i][j] = str(3 * i + j + 1)
                    best_score = min(score, best_score)
        return best_score

def best_move(board):
    best_score = -float('inf')
    move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] not in ['X', 'O']:
                board[i][j] = 'X'
                score = minimax(board, False)
                board[i][j] = str(3 * i + j + 1)
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def play_game():
    board = [[str(3 * i + j + 1) for j in range(3)] for i in range(3)]
    print("Tic-Tac-Toe\nYou are O, AI is X")

    while True:
        print_board(board)
        move = input("Enter your move (1-9): ")
        if not move.isdigit():
            print("Invalid input.")
            continue
        move = int(move)
        r, c = (move - 1) // 3, (move - 1) % 3
        if board[r][c] in ['X', 'O']:
            print("Spot taken.")
            continue
        board[r][c] = 'O'

        if evaluate(board) is not None:
            break

        global nodes_examined
        nodes_examined = 0
        ai_r, ai_c = best_move(board)
        board[ai_r][ai_c] = 'X'
        print(f"AI chose move {(ai_r * 3 + ai_c + 1)}")
        print(f"Nodes examined: {nodes_examined}")

        if evaluate(board) is not None:
            break

    print_board(board)
    result = evaluate(board)
    if result == 1:
        print("AI wins!")
    elif result == -1:
        print("You win!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()
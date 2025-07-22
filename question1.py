from time import time

def solve_n_queens(n):
    """
    Solves the N-Queens problem using backtracking.
    Returns a list of all valid solutions. Each solution is a list of column positions,
    where the index represents the row and the value represents the column of the queen.
    """
    solutions = []
    board = [-1] * n  # board[i] = column position of queen in row i

    def is_safe(row, col):
        """
        Checks whether placing a queen at (row, col) is safe.
        A queen is safe if there is no other queen in the same column or on the same diagonal.
        """
        for prev_row in range(row):
            if (board[prev_row] == col or
                abs(board[prev_row] - col) == abs(prev_row - row)):
                return False
        return True

    def backtrack(row):
        """
        Tries to place a queen in each column of the current row.
        If a safe spot is found, places the queen and moves to the next row recursively.
        If the last row is reached, saves the current board configuration.
        """
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1

    backtrack(0)
    return solutions

def print_solutions(solutions):
    """
    Prints each N-Queens solution in a readable board format.
    'Q' represents a queen, and '.' represents an empty space.
    """
    for idx, solution in enumerate(solutions):
        print(f"Solution {idx + 1}:")
        n = len(solution)
        for row in range(n):
            line = ['.'] * n
            line[solution[row]] = 'Q'
            print(' '.join(line))
        print()  # Blank line between solutions

if __name__ == "__main__":
    # Test with 8 queens
    # start = time()
    # solutions_8 = solve_n_queens(8)
    # end = time()
    # print(f"8-Queens: Found {len(solutions_8)} solution(s) in {end - start:.4f} seconds")
    # # Uncomment to view all solutions
    # print_solutions(solutions_8)
    # print("=" * 20)

    # Test with 100 queens
    start = time()
    solutions_100 = solve_n_queens(100)
    end = time()
    print(f"100-Queens: Found {len(solutions_100)} solution(s) in {end - start:.4f} seconds")
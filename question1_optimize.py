from time import time
def solve_n_queens_large(n):
    solution = []
    # Place even numbers first, then odd numbers
    for i in range(2, n + 1, 2):
        solution.append(i)
    for i in range(1, n + 1, 2):
        solution.append(i)
    return solution

if __name__ == "__main__":
    # n = 100
    # start = time()
    # solution = solve_n_queens_large(n)
    # end = time()
    # print(f"Solution for {n} queens: {solution}")
    # print(f"Time taken: {end - start:.4f} seconds")

    print("=" * 20)
    n1 = 100000
    start = time()
    solution = solve_n_queens_large(n1)
    end = time()
    print(f"Solution for {n1} queens: {solution}")
    print(f"Time taken: {end - start:.4f} seconds")
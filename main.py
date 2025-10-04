from NonogramSolver import NonogramSolver

def main():
    rows = [
        [4],
        [3, 2],
        [2, 2, 3],
        [3, 5],
        [3],
        [2, 2],
        [1],
        [5, 1],
        [2, 3],
        [2],
    ]

    cols = [
        [1, 1, 3],
        [1, 2, 2],
        [2, 1, 1],
        [5, 1],
        [2, 1, 1],
        [3, 2],
        [2, 1, 2],
        [2, 1],
        [3, 1],
        [1, 1, 1, 1],
    ]

    solver = NonogramSolver(rows, cols)
    sol = solver.solve()
    print(solver.render() if sol else "No solution found.")

if __name__ == "__main__":
    main()

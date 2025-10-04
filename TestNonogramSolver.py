import unittest
from NonogramSolver import NonogramSolver, FILLED, EMPTY

class TestNonogramSolver(unittest.TestCase):

    # -------------------------------------------------------------
    # 1. CORE SOLVER FUNCTIONALITY (end-to-end behavior)
    # -------------------------------------------------------------

    def test_simple_5x5_plus(self):
        """Verify solver can handle a small valid 5x5 plus-sign puzzle."""
        rows = [[1], [1], [5], [1], [1]]
        cols = [[1], [1], [5], [1], [1]]
        solver = NonogramSolver(rows, cols)
        sol = solver.solve()
        self.assertIsNotNone(sol)
        expected = [
            [EMPTY, EMPTY, FILLED, EMPTY, EMPTY],
            [EMPTY, EMPTY, FILLED, EMPTY, EMPTY],
            [FILLED, FILLED, FILLED, FILLED, FILLED],
            [EMPTY, EMPTY, FILLED, EMPTY, EMPTY],
            [EMPTY, EMPTY, FILLED, EMPTY, EMPTY],
        ]
        self.assertEqual(sol, expected)

    def test_contradictory_clues(self):
        """Unsolvable puzzle should return None."""
        rows = [[4]]
        cols = [[1], [1], [1]]
        solver = NonogramSolver(rows, cols)
        result = solver.solve()
        self.assertIsNone(result)

    def test_branching_required(self):
        """Ensure DFS branching works when guessing is required."""
        rows = [[1], [1]]
        cols = [[1], [1]]
        solver = NonogramSolver(rows, cols)
        sol = solver.solve()
        self.assertIsNotNone(sol)

    def test_recursive_branch_failure(self):
        """Ensure solver returns None when all DFS branches fail."""
        rows = [[2], [2]]
        cols = [[1], [1]]
        solver = NonogramSolver(rows, cols)
        result = solver.solve()
        self.assertIsNone(result)

    # -------------------------------------------------------------
    # 2. LINE DEDUCTION AND PATTERN LOGIC
    # -------------------------------------------------------------

    def test_envelope_deduction_basic(self):
        """Test overlap rule in envelope deduction."""
        solver = NonogramSolver([], [])
        line = [-1] * 10
        runs = (7,)
        ded, _ = solver._deduce_line_envelope(line, runs)
        filled_indices = [i for i, x in enumerate(ded) if x == FILLED]
        self.assertEqual(filled_indices, [3, 4, 5, 6])

    def test_envelope_contradiction(self):
        """Envelope deduction should raise contradiction for impossible line."""
        solver = NonogramSolver([], [])
        line = [1, 0, 0]  # filled but empty clue
        with self.assertRaises(ValueError):
            solver._deduce_line_envelope(line, ())

    def test_deduce_line_envelope_contradiction(self):
        """Explicit contradiction with filled cell not coverable."""
        solver = NonogramSolver([], [])
        line = [1, 0, 0]
        with self.assertRaises(ValueError):
            solver._deduce_line_envelope(line, ())

    def test_gen_all_line_patterns(self):
        """Pattern generator should enumerate correct number of patterns."""
        solver = NonogramSolver([], [])
        patterns = solver._gen_all_line_patterns(5, (2,))
        self.assertEqual(len(patterns), 4)

    def test_fits_line(self):
        """Check compatibility between partial line and full pattern."""
        solver = NonogramSolver([], [])
        self.assertTrue(solver._fits_line([-1, 1, 0], (0, 1, 0)))
        self.assertFalse(solver._fits_line([1, 0, 0], (0, 1, 0)))

    def test_line_matches(self):
        """Ensure filled sequences correctly match clues."""
        solver = NonogramSolver([], [])
        self.assertTrue(solver._line_matches([1, 1, 0, 1, 0], (2, 1)))
        self.assertFalse(solver._line_matches([1, 0, 1, 1], (1, 1)))

    def test_earliest_and_latest_starts(self):
        """Verify earliest and latest block placements are computed."""
        solver = NonogramSolver([], [])
        line = [-1] * 10
        runs = (3, 2)
        earliest = solver._earliest_starts(line, runs)
        latest = solver._latest_starts(line, runs)
        self.assertIsInstance(earliest, list)
        self.assertIsInstance(latest, list)

    # -------------------------------------------------------------
    # 3. UTILITY AND HELPER METHODS
    # -------------------------------------------------------------

    def test_row_and_col_helpers(self):
        solver = NonogramSolver([[1]], [[1]])
        solver.grid = [[1]]
        self.assertEqual(solver.row(0), [1])
        self.assertEqual(solver.col(0), [1])

    def test_set_row_and_set_col(self):
        solver = NonogramSolver([[1, 1]], [[1], [1], [1]])
        solver.grid = [[-1, -1, -1]]
        solver.set_row(0, [1, 0, 1])
        self.assertEqual(solver.row(0), [1, 0, 1])
        solver.set_col(1, [0])
        self.assertEqual(solver.col(1), [0])

    def test_clone_grid_independence(self):
        solver = NonogramSolver([[1]], [[1]])
        clone = solver._clone()
        solver.grid[0][0] = 1
        self.assertNotEqual(solver.grid, clone.grid)

    def test_render_output(self):
        solver = NonogramSolver([[1]], [[1]])
        solver.grid = [[1, 0, -1]]
        solver.R, solver.C = 1, 3
        result = solver.render(filled_char="#", empty_char=".")
        self.assertEqual(result, "#.?")

    # -------------------------------------------------------------
    # 4. EDGE CASES AND INTERNAL LOGIC COVERAGE
    # -------------------------------------------------------------

    def test_propagate_runs_without_error(self):
        solver = NonogramSolver([[1]], [[1]])
        solver._propagate()  # should not raise

    def test_propagate_no_change(self):
        solver = NonogramSolver([[1]], [[1]])
        solver.grid = [[1]]
        solver._propagate()
        self.assertTrue(solver.is_solved())

    def test_reduce_once_contradiction(self):
        """Force ValueError during envelope deduction (impossible clues)."""
        # 1x1 grid, but row expects a block of length 2 → impossible
        rows = [[2]]  # impossible for 1-cell line
        cols = [[1]]
        solver = NonogramSolver(rows, cols)
        solver.grid = [[-1]]  # still unknown (forces deduction attempt)
        with self.assertRaises(ValueError):
            solver._reduce_once()

    def test_select_branch_line(self):
        solver = NonogramSolver([[1], [1]], [[1], [1]])
        solver.grid = [[-1, -1], [-1, -1]]
        is_row, idx, pats = solver._select_branch_line()
        self.assertIn(is_row, [True, False])
        self.assertIsInstance(pats, list)

    def test_select_branch_line_tiebreak(self):
        solver = NonogramSolver([[1], [1]], [[1], [1]])
        solver.grid = [[-1, -1], [-1, -1]]
        is_row, idx, pats = solver._select_branch_line()
        self.assertIn(is_row, [True, False])
        self.assertIsInstance(pats, list)


if __name__ == "__main__":
    unittest.main()

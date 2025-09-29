from functools import lru_cache
from typing import List, Tuple, Optional

# Cell states: -1 unknown, 0 empty, 1 filled
UNKNOWN, EMPTY, FILLED = -1, 0, 1

class NonogramSolver:
    def __init__(self, row_clues: List[List[int]], col_clues: List[List[int]]):
        self.R = len(row_clues)
        self.C = len(col_clues)
        self.row_clues = [tuple(c) for c in row_clues]
        self.col_clues = [tuple(c) for c in col_clues]
        self.grid = [[UNKNOWN for _ in range(self.C)] for _ in range(self.R)]

    # -------------------- Utilities --------------------
    def row(self, r): return self.grid[r]
    def col(self, c): return [self.grid[r][c] for r in range(self.R)]
    def set_row(self, r, vals):
        for c, v in enumerate(vals):
            if v != UNKNOWN: self.grid[r][c] = v
    def set_col(self, c, vals):
        for r, v in enumerate(vals):
            if v != UNKNOWN: self.grid[r][c] = v

    @staticmethod
    def _fits_line(line: List[int], candidate: Tuple[int, ...]) -> bool:
        # line contains UNKNOWN/EMPTY/FILLED; candidate is 0/1 tuple
        for a, b in zip(line, candidate):
            if a != UNKNOWN and a != b:
                return False
        return True

    @staticmethod
    def _intersect_patterns(patterns: List[Tuple[int, ...]]) -> List[int]:
        if not patterns:
            return []
        L = len(patterns[0])
        out = []
        for i in range(L):
            col = {p[i] for p in patterns}
            if len(col) == 1:
                out.append(col.pop())
            else:
                out.append(UNKNOWN)
        return out

    # -------------------- Line generator --------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _gen_all_line_patterns(length: int, clues: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Generate all 0/1 patterns of given length matching clues (no constraints from current cells).
        Uses simple recursion + spacing rules.
        """
        if not clues:
            return [tuple([EMPTY]*length)]
        k = clues[0]
        rest = clues[1:]
        patterns = []
        # Minimum space needed for remaining blocks (including separators)
        min_rest = sum(rest) + max(0, len(rest) - 1)
        # Try placing first block starting at each feasible start position
        for start in range(0, length - k - min_rest + 1):
            # Leading empties until 'start'
            prefix = [EMPTY]*start + [FILLED]*k
            if rest:
                # Must have one EMPTY separator
                if start + k >= length:
                    continue
                prefix = prefix + [EMPTY]
                tail_len = length - len(prefix)
                for tail in NonogramSolver._gen_all_line_patterns(tail_len, rest):
                    patterns.append(tuple(prefix + list(tail)))
            else:
                # No more blocks; fill remaining with EMPTY
                suffix = [EMPTY]*(length - len(prefix))
                patterns.append(tuple(prefix + suffix))
        return patterns

    def _compatible_patterns(self, line: List[int], clues: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Filter pre-generated patterns by current line constraints to avoid re-enumerating."""
        all_patterns = self._gen_all_line_patterns(len(line), clues)
        # Quick prune: remove any pattern conflicting with fixed cells
        return [p for p in all_patterns if self._fits_line(line, p)]

    # -------------------- Constraint propagation --------------------
    def _reduce_once(self) -> bool:
        """
        One sweep over all rows and columns:
          - Compute compatible patterns
          - Intersect to deduce forced cells
        Returns True if any cell changed.
        """
        changed = False

        # Rows
        for r in range(self.R):
            line = self.row(r)
            if UNKNOWN not in line:
                continue
            pats = self._compatible_patterns(line, self.row_clues[r])
            if not pats:
                # Infeasible state
                raise ValueError("Contradiction in row {}".format(r))
            deduced = self._intersect_patterns(pats)
            for c, v in enumerate(deduced):
                if v != UNKNOWN and self.grid[r][c] == UNKNOWN:
                    self.grid[r][c] = v
                    changed = True

        # Columns
        for c in range(self.C):
            line = self.col(c)
            if UNKNOWN not in line:
                continue
            pats = self._compatible_patterns(line, self.col_clues[c])
            if not pats:
                raise ValueError("Contradiction in col {}".format(c))
            deduced = self._intersect_patterns(pats)
            for r, v in enumerate(deduced):
                if v != UNKNOWN and self.grid[r][c] == UNKNOWN:
                    self.grid[r][c] = v
                    changed = True

        return changed

    def _propagate(self):
        while True:
            if not self._reduce_once():
                return

    # -------------------- Checkers --------------------
    def is_solved(self) -> bool:
        return all(UNKNOWN not in row for row in self.grid) and self._all_lines_valid()

    def _line_matches(self, line: List[int], clues: Tuple[int, ...]) -> bool:
        # Convert a 0/1/UNKNOWN line to clue sequence (fail if UNKNOWN present)
        if UNKNOWN in line:
            return False
        blocks = []
        count = 0
        for x in line:
            if x == FILLED:
                count += 1
            else:
                if count > 0:
                    blocks.append(count)
                    count = 0
        if count > 0:
            blocks.append(count)
        return tuple(blocks) == clues

    def _all_lines_valid(self) -> bool:
        for r in range(self.R):
            if not self._line_matches(self.row(r), self.row_clues[r]):
                return False
        for c in range(self.C):
            if not self._line_matches(self.col(c), self.col_clues[c]):
                return False
        return True

    # -------------------- Search (backtracking) --------------------
    def _select_branch_line(self) -> Tuple[bool, bool, int, List[Tuple[int, ...]]]:
        """
        Choose a line (row or column) with UNKNOWNs and minimal number of compatible patterns (>1).
        Returns (is_row, is_col, index, patterns) where one of is_row/is_col is True.
        If no branching needed, returns (False, False, -1, []).
        """
        best = (None, None, None)   # (is_row, index, pats)
        best_count = float('inf')

        # Rows
        for r in range(self.R):
            line = self.row(r)
            if UNKNOWN not in line:
                continue
            pats = self._compatible_patterns(line, self.row_clues[r])
            if len(pats) == 1:
                # Will be set by propagation next pass; skip as branch
                continue
            if 1 < len(pats) < best_count:
                best = (True, r, pats)
                best_count = len(pats)
                if best_count == 2:
                    break  # very constrained, good enough

        # Columns (only search if we didn't find a 2-option row)
        if best_count > 2:
            for c in range(self.C):
                line = self.col(c)
                if UNKNOWN not in line:
                    continue
                pats = self._compatible_patterns(line, self.col_clues[c])
                if len(pats) == 1:
                    continue
                if 1 < len(pats) < best_count:
                    best = (False, c, pats)
                    best_count = len(pats)
                    if best_count == 2:
                        break

        if best[0] is None:
            return (False, False, -1, [])
        is_row, idx, pats = best
        return (is_row, not is_row, idx, pats)

    def _clone(self):
        other = NonogramSolver(self.row_clues, self.col_clues)
        other.grid = [row[:] for row in self.grid]
        return other

    def solve(self) -> Optional[List[List[int]]]:
        try:
            self._propagate()
        except ValueError:
            return None  # immediate contradiction

        if self.is_solved():
            return self.grid

        # Pick a constrained line and branch
        is_row, is_col, idx, pats = self._select_branch_line()
        if idx == -1:
            # No unknowns left but not matching clues (rare), or stuck without choices
            return None

        if is_row:
            current = self.row(idx)
            for p in pats:
                if not self._fits_line(current, p):
                    continue
                child = self._clone()
                child.set_row(idx, p)
                res = child.solve()
                if res is not None:
                    self.grid = res
                    return res
        else:
            current = self.col(idx)
            for p in pats:
                if not self._fits_line(current, p):
                    continue
                child = self._clone()
                child.set_col(idx, p)
                res = child.solve()
                if res is not None:
                    self.grid = res
                    return res

        return None

    # Print
    def render(self, filled_char="█", empty_char="·") -> str:
        s = []
        for r in range(self.R):
            row = "".join(filled_char if x == FILLED else empty_char if x == EMPTY else "?" for x in self.grid[r])
            s.append(row)
        return "\n".join(s)


if __name__ == "__main__":
    rows = [
        [4],
        [2, 2, 1],
        [3, 2],
        [1],
        [2, 2],
        [1, 2],
        [2, 3],
        [1, 1],
        [7],
        [2, 5],
    ]

    cols = [
        [1, 2, 2,1],
        [1, 1, 1, 1],
        [3,1],
        [2, 1],
        [1, 1, 1],
        [1, 4],
        [1, 2, 2],
        [1, 1, 2],
        [1, 1, 3],
        [2, 1, 1],
    ]

    solver = NonogramSolver(rows, cols)
    solved = solver.solve()
    if solved is None:
        print("No solution.")
    else:
        print(solver.render())

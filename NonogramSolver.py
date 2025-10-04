from functools import lru_cache
from typing import List, Tuple, Optional

# Cell States
UNKNOWN, EMPTY, FILLED = -1, 0, 1

class NonogramSolver:
    """
    Initialize the solver with row and column clues.

    Args:
        row_clues: A list of lists, each sublist containing integers for the row constraints.
        col_clues: A list of lists, each sublist containing integers for the column constraints.
    """
    def __init__(self, row_clues: List[List[int]], col_clues: List[List[int]]):
        self.R = len(row_clues)
        self.C = len(col_clues)
        self.row_clues = [tuple(c) for c in row_clues]
        self.col_clues = [tuple(c) for c in col_clues]
        self.grid = [[UNKNOWN for _ in range(self.C)] for _ in range(self.R)]

    def row(self, r): return self.grid[r]

    def col(self, c): return [self.grid[r][c] for r in range(self.R)]

    def set_row(self, r, vals):
        for c, v in enumerate(vals):
            if v != UNKNOWN:
                self.grid[r][c] = v

    def set_col(self, c, vals):
        for r, v in enumerate(vals):
            if v != UNKNOWN:
                self.grid[r][c] = v

    @staticmethod
    def _min_rest(runs: Tuple[int, ...], i: int) -> int:
        """Return the minimum length required to fit remaining blocks after block `i`."""
        rest = runs[i+1:]
        return sum(rest) + max(0, len(rest) - 1)

    @staticmethod
    def _can_place_block(line: List[int], s: int, k: int, is_last: bool) -> bool:
        """
        Check if a block of size `k` can be placed starting at index `s` on the given line.

        Ensures it does not overlap EMPTY cells or FILLED cells outside its range.
        """
        L = len(line)
        if s > 0 and line[s-1] == FILLED:
            return False
        if s + k < L and line[s + k] == FILLED:
            return False
        for t in range(s, s + k):
            if line[t] == EMPTY:
                return False
        return True

    @staticmethod
    def _earliest_starts(line: List[int], runs: Tuple[int, ...]) -> List[int]:
        """
        Compute the earliest possible start index for each block on this line.

        Returns:
            A list of start indices for each block.
        Raises:
            ValueError: if any block cannot legally fit.
        """
        L = len(line)
        m = len(runs)
        starts = [None] * m
        pos = 0
        for i, k in enumerate(runs):
            limit = L - k - NonogramSolver._min_rest(runs, i)
            s = pos
            placed = False
            while s <= limit:
                if NonogramSolver._can_place_block(line, s, k, is_last=(i == m - 1)):
                    placed = True
                    break
                s += 1
            if not placed:
                raise ValueError("No earliest placement for block {} of size {}".format(i, k))
            starts[i] = s
            pos = s + k + 1
        return starts  # type: ignore

    @staticmethod
    def _latest_starts(line: List[int], runs: Tuple[int, ...]) -> List[int]:
        """
         Compute the latest possible start index for each block (reverse envelope reasoning).
        """
        L = len(line)
        if not runs:
            return []
        rline = list(reversed(line))
        rruns = tuple(reversed(runs))
        estarts_rev = NonogramSolver._earliest_starts(rline, rruns)
        m = len(runs)
        latest = [None] * m
        for i, k in enumerate(runs):
            j = m - 1 - i
            s_rev = estarts_rev[j]
            latest[i] = L - (s_rev + k)
        return latest  # type: ignore

    @staticmethod
    def _deduce_line_envelope(line: List[int], runs: Tuple[int, ...]) -> Tuple[List[int], bool]:
        """
        Apply envelope (overlap) deduction to a single line.

        Returns:
            A tuple (deductions, changed):
                - deductions: list of new cell states (-1, 0, 1)
                - changed: True if any cell was newly deduced
        Raises:
            ValueError: if a contradiction is found (impossible arrangement).
        """
        L = len(line)
        m = len(runs)
        if m == 0:
            ded = [EMPTY] * L
            for i, v in enumerate(line):
                if v == FILLED:
                    raise ValueError("Contradiction: line has FILLED but clues are empty")
            changed = any(line[i] != ded[i] for i in range(L))
            return ded, changed

        estarts = NonogramSolver._earliest_starts(line, runs)
        lstarts = NonogramSolver._latest_starts(line, runs)

        for i in range(m):
            if estarts[i] > lstarts[i]:
                raise ValueError("Contradiction: block {} cannot be placed".format(i))

        deduced = [UNKNOWN] * L
        covered_any = [False] * L

        for i, k in enumerate(runs):
            e = estarts[i]
            l = lstarts[i]
            overlap_start = l
            overlap_end = e + k - 1
            if overlap_start <= overlap_end:
                for x in range(overlap_start, overlap_end + 1):
                    deduced[x] = FILLED
            for s in range(e, l + 1):
                if NonogramSolver._can_place_block(line, s, k, is_last=(i == m - 1)):
                    for x in range(s, s + k):
                        covered_any[x] = True

        for x in range(L):
            if deduced[x] == UNKNOWN and not covered_any[x]:
                deduced[x] = EMPTY

        for x, v in enumerate(line):
            if v == FILLED and not covered_any[x]:
                raise ValueError("Contradiction: known FILLED at {} cannot be covered".format(x))

        changed = False
        out = [UNKNOWN] * L
        for i in range(L):
            if deduced[i] != UNKNOWN and line[i] == UNKNOWN:
                out[i] = deduced[i]
                changed = True
            else:
                out[i] = UNKNOWN
        return out, changed

    @staticmethod
    @lru_cache(maxsize=None)
    def _gen_all_line_patterns(length: int, clues: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
         Generate all possible line patterns for a given clue and length.

         Cached with @lru_cache to avoid re-computation.
         """
        if not clues:
            return [tuple([EMPTY] * length)]
        k = clues[0]
        rest = clues[1:]
        patterns = []
        min_rest = sum(rest) + max(0, len(rest) - 1)
        for start in range(0, length - k - min_rest + 1):
            prefix = [EMPTY] * start + [FILLED] * k
            if rest:
                if start + k >= length:
                    continue
                prefix = prefix + [EMPTY]
                tail_len = length - len(prefix)
                for tail in NonogramSolver._gen_all_line_patterns(tail_len, rest):
                    patterns.append(tuple(prefix + list(tail)))
            else:
                suffix = [EMPTY] * (length - len(prefix))
                patterns.append(tuple(prefix + suffix))
        return patterns

    def _fits_line(self, line: List[int], candidate: Tuple[int, ...]) -> bool:
        """Return True if a candidate pattern matches a partially known line."""
        return all(a == UNKNOWN or a == b for a, b in zip(line, candidate))

    def _compatible_patterns(self, line: List[int], clues: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Return all candidate patterns that match a given partially filled line."""
        allp = self._gen_all_line_patterns(len(line), clues)
        return [p for p in allp if self._fits_line(line, p)]

    def _reduce_once(self) -> bool:
        """
        Perform one round of logical deductions across all rows and columns.

        Returns:
            True if any new cell was deduced; False otherwise.
        Raises:
            ValueError: if a contradiction is found.
        """
        changed = False

        for r in range(self.R):
            line = self.row(r)
            if UNKNOWN in line:
                ded, ch = self._deduce_line_envelope(line, self.row_clues[r])
                if ch:
                    for c, v in enumerate(ded):
                        if v != UNKNOWN and self.grid[r][c] == UNKNOWN:
                            self.grid[r][c] = v
                            changed = True

        for c in range(self.C):
            line = self.col(c)
            if UNKNOWN in line:
                ded, ch = self._deduce_line_envelope(line, self.col_clues[c])
                if ch:
                    for r, v in enumerate(ded):
                        if v != UNKNOWN and self.grid[r][c] == UNKNOWN:
                            self.grid[r][c] = v
                            changed = True

        for r in range(self.R):
            line = self.row(r)
            if UNKNOWN not in line:
                continue
            pats = self._compatible_patterns(line, self.row_clues[r])
            if not pats:
                raise ValueError("Contradiction in row {}".format(r))
            L = len(line)
            for i in range(L):
                vals = {p[i] for p in pats}
                if len(vals) == 1 and self.grid[r][i] == UNKNOWN:
                    self.grid[r][i] = vals.pop()
                    changed = True

        for c in range(self.C):
            line = self.col(c)
            if UNKNOWN not in line:
                continue
            pats = self._compatible_patterns(line, self.col_clues[c])
            if not pats:
                raise ValueError("Contradiction in col {}".format(c))
            L = len(line)
            for i in range(L):
                vals = {p[i] for p in pats}
                if len(vals) == 1 and self.grid[i][c] == UNKNOWN:
                    self.grid[i][c] = vals.pop()
                    changed = True

        return changed

    def _propagate(self):
        """Run logical deductions repeatedly until no new cells can be filled."""
        while True:
            if not self._reduce_once():
                return

    def _line_matches(self, line: List[int], clues: Tuple[int, ...]) -> bool:
        if UNKNOWN in line:
            return False
        blocks = []
        cnt = 0
        for x in line:
            if x == FILLED:
                cnt += 1
            else:
                if cnt > 0:
                    blocks.append(cnt)
                    cnt = 0
        if cnt > 0:
            blocks.append(cnt)
        return tuple(blocks) == clues

    def _all_lines_valid(self) -> bool:
        """Return True if all rows and columns match their clues."""
        for r in range(self.R):
            if not self._line_matches(self.row(r), self.row_clues[r]):
                return False
        for c in range(self.C):
            if not self._line_matches(self.col(c), self.col_clues[c]):
                return False
        return True

    def is_solved(self) -> bool:
        """Check if puzzle is completely filled and all clues are satisfied."""
        return all(UNKNOWN not in row for row in self.grid) and self._all_lines_valid()

    def _select_branch_line(self) -> Tuple[bool, int, List[Tuple[int, ...]]]:
        """
        Choose the next row or column to branch on using MRV (minimum remaining values).

        Returns:
            (is_row, index, patterns) where:
                - is_row: True if branching on a row, False for column
                - index: line index
                - patterns: list of possible patterns
        """
        best_idx = -1
        best_pats = None
        best_is_row = True
        best_count = float('inf')

        for r in range(self.R):
            line = self.row(r)
            if UNKNOWN not in line:
                continue
            pats = self._compatible_patterns(line, self.row_clues[r])
            if 1 < len(pats) < best_count:
                best_count = len(pats)
                best_is_row = True
                best_idx = r
                best_pats = pats
                if best_count == 2:
                    break

        if best_count > 2:
            for c in range(self.C):
                line = self.col(c)
                if UNKNOWN not in line:
                    continue
                pats = self._compatible_patterns(line, self.col_clues[c])
                if 1 < len(pats) < best_count:
                    best_count = len(pats)
                    best_is_row = False
                    best_idx = c
                    best_pats = pats
                    if best_count == 2:
                        break

        if best_idx == -1:
            return False, -1, []
        return best_is_row, best_idx, best_pats  # type: ignore

    def _clone(self):
        """Create a deep copy of the current solver state (grid + clues)."""
        other = NonogramSolver(self.row_clues, self.col_clues)
        other.grid = [row[:] for row in self.grid]
        return other

    def solve(self) -> Optional[List[List[int]]]:
        """
        Solve the Nonogram puzzle.

        The method alternates between logic propagation and DFS guessing.

        Returns:
            A 2D list representing the solved grid (0/1 values),
            or None if the puzzle has no valid solution.
        """
        try:
            self._propagate()
        except ValueError:
            return None

        if self.is_solved():
            return self.grid

        is_row, idx, pats = self._select_branch_line()
        if idx == -1:
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

    def render(self, filled_char="█", empty_char="·") -> str:
        """
        Print current grid as a string.

        Args:
            filled_char: Character used for filled cells.
            empty_char: Character used for empty cells.
        Returns:
            A string representation of the current grid.
        """
        s = []
        for r in range(self.R):
            row = "".join(filled_char if x == FILLED else empty_char if x == EMPTY else "?" for x in self.grid[r])
            s.append(row)
        return "\n".join(s)


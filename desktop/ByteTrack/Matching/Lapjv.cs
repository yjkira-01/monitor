namespace ByteTrack.Matching;

/// <summary>
/// Jonker–Volgenant linear assignment, matching the semantics of
/// <c>lap.lapjv(cost, extend_cost=True, cost_limit=thresh)</c> used in ByteTrack.
/// </summary>
/// <remarks>
/// The core dense solver is a C# port of the classic LAPJV algorithm (Jonker &amp;
/// Volgenant, 1987). Rectangular inputs and a cost limit are handled exactly as
/// <c>lap.lapjv</c> does: the matrix is squared out with a large "virtual" cost,
/// entries above <c>costLimit</c> are replaced by that large cost so
/// they are never chosen, and any assignment landing on a virtual/over-limit
/// cell is reported as unmatched (index -1).
/// </remarks>
public static class Lapjv
{
    private const double Large = 1e15;

    /// <summary>
    /// Solves the assignment problem for a rectangular cost matrix.
    /// </summary>
    /// <param name="cost">Row-major cost matrix (<c>nRows × nCols</c>).</param>
    /// <param name="costLimit">
    /// Upper bound: pairs with cost &gt; this are forbidden. Use
    /// <see cref="double.PositiveInfinity"/> for no limit.
    /// </param>
    /// <returns>
    /// <c>rowToCol[i]</c> = assigned column for row i (or -1), and
    /// <c>colToRow[j]</c> = assigned row for column j (or -1).
    /// </returns>
    public static (int[] RowToCol, int[] ColToRow) Solve(double[,] cost, double costLimit)
    {
        int nRows = cost.GetLength(0);
        int nCols = cost.GetLength(1);

        if (nRows == 0 || nCols == 0)
        {
            var emptyRows = new int[nRows];
            var emptyCols = new int[nCols];
            Array.Fill(emptyRows, -1);
            Array.Fill(emptyCols, -1);
            return (emptyRows, emptyCols);
        }

        // extend_cost=True: build a square matrix of size n = nRows + nCols.
        // Real block [0..nRows, 0..nCols] carries the (limited) costs; the rest
        // is filled with a large virtual cost so extra rows/cols can always be
        // matched among themselves without disturbing the real optimum.
        bool hasLimit = !double.IsPositiveInfinity(costLimit);

        // Determine the virtual cost. lap uses cost_limit / 2 when a limit is
        // given, otherwise (max finite cost + 1). This keeps virtual matches
        // cheaper than any forbidden real cell, so real cells over the limit are
        // rejected in favour of a virtual (⇒ unmatched) assignment.
        double virtualCost;
        if (hasLimit)
        {
            virtualCost = costLimit / 2.0;
        }
        else
        {
            double maxFinite = 0.0;
            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    double c = cost[i, j];
                    if (!double.IsInfinity(c) && c > maxFinite)
                    {
                        maxFinite = c;
                    }
                }
            }

            virtualCost = maxFinite + 1.0;
        }

        int n = nRows + nCols;
        var square = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i < nRows && j < nCols)
                {
                    double c = cost[i, j];
                    if (double.IsInfinity(c) || (hasLimit && c > costLimit))
                    {
                        square[i, j] = Large;
                    }
                    else
                    {
                        square[i, j] = c;
                    }
                }
                else
                {
                    square[i, j] = virtualCost;
                }
            }
        }

        int[] rowSol = SolveSquare(square, n);

        var rowToCol = new int[nRows];
        var colToRow = new int[nCols];
        Array.Fill(rowToCol, -1);
        Array.Fill(colToRow, -1);

        for (int i = 0; i < nRows; i++)
        {
            int j = rowSol[i];
            // Accept only assignments to real columns whose cost is admissible.
            if (j < nCols)
            {
                double c = cost[i, j];
                if (!double.IsInfinity(c) && !(hasLimit && c > costLimit))
                {
                    rowToCol[i] = j;
                    colToRow[j] = i;
                }
            }
        }

        return (rowToCol, colToRow);
    }

    /// <summary>
    /// Dense LAPJV for a square <paramref name="n"/>×<paramref name="n"/> cost
    /// matrix. Returns <c>rowSol[i]</c> = column assigned to row i.
    /// </summary>
    private static int[] SolveSquare(double[,] cost, int n)
    {
        var rowSol = new int[n];   // row -> col
        var colSol = new int[n];   // col -> row
        var y = new double[n];     // dual variables on columns (v)
        var x = new double[n];     // dual variables on rows (u)
        Array.Fill(rowSol, -1);
        Array.Fill(colSol, -1);

        // --- Column reduction ---
        var free = new int[n];
        int numFree = 0;
        var matchedRow = new bool[n];

        for (int j = n - 1; j >= 0; j--)
        {
            double min = cost[0, j];
            int imin = 0;
            for (int i = 1; i < n; i++)
            {
                if (cost[i, j] < min)
                {
                    min = cost[i, j];
                    imin = i;
                }
            }

            y[j] = min;
            if (!matchedRow[imin])
            {
                matchedRow[imin] = true;
                colSol[j] = imin;
                rowSol[imin] = j;
            }
            else
            {
                colSol[j] = -1;
            }
        }

        // --- Reduction transfer + collect free rows ---
        for (int i = 0; i < n; i++)
        {
            if (rowSol[i] == -1)
            {
                free[numFree++] = i;
            }
            else
            {
                int j1 = rowSol[i];
                double min = double.PositiveInfinity;
                for (int j = 0; j < n; j++)
                {
                    if (j != j1 && cost[i, j] - y[j] < min)
                    {
                        min = cost[i, j] - y[j];
                    }
                }

                x[i] = min;
            }
        }

        // --- Augmenting row reduction (2 passes) ---
        for (int loop = 0; loop < 2 && numFree > 0; loop++)
        {
            int k = 0;
            int prevNumFree = numFree;
            numFree = 0;
            while (k < prevNumFree)
            {
                int i = free[k++];
                double umin = cost[i, 0] - y[0];
                int j1 = 0;
                double usubmin = double.PositiveInfinity;
                int j2 = -1;
                for (int j = 1; j < n; j++)
                {
                    double h = cost[i, j] - y[j];
                    if (h < usubmin)
                    {
                        if (h >= umin)
                        {
                            usubmin = h;
                            j2 = j;
                        }
                        else
                        {
                            usubmin = umin;
                            umin = h;
                            j2 = j1;
                            j1 = j;
                        }
                    }
                }

                int i0 = colSol[j1];
                if (umin < usubmin)
                {
                    y[j1] -= usubmin - umin;
                }
                else if (i0 >= 0)
                {
                    j1 = j2;
                    i0 = colSol[j2];
                }

                rowSol[i] = j1;
                colSol[j1] = i;
                if (i0 >= 0)
                {
                    if (umin < usubmin)
                    {
                        free[--k] = i0;
                    }
                    else
                    {
                        free[numFree++] = i0;
                    }

                    rowSol[i0] = -1;
                }
            }
        }

        // --- Augmentation (shortest augmenting path via Dijkstra) ---
        var d = new double[n];
        var pred = new int[n];
        var colList = new int[n];

        for (int f = 0; f < numFree; f++)
        {
            int freeRow = free[f];
            for (int j = 0; j < n; j++)
            {
                d[j] = cost[freeRow, j] - y[j];
                pred[j] = freeRow;
                colList[j] = j;
            }

            int low = 0;
            int up = 0;
            bool unassignedFound = false;
            int last = -1;
            int endCol = -1;
            double minD = 0.0;

            while (!unassignedFound)
            {
                if (up == low)
                {
                    last = low - 1;
                    minD = d[colList[up++]];
                    for (int k = up; k < n; k++)
                    {
                        int j = colList[k];
                        double h = d[j];
                        if (h <= minD)
                        {
                            if (h < minD)
                            {
                                up = low;
                                minD = h;
                            }

                            colList[k] = colList[up];
                            colList[up++] = j;
                        }
                    }

                    for (int k = low; k < up; k++)
                    {
                        int j = colList[k];
                        if (colSol[j] < 0)
                        {
                            endCol = j;
                            unassignedFound = true;
                            break;
                        }
                    }
                }

                if (!unassignedFound)
                {
                    int j1 = colList[low++];
                    int i = colSol[j1];
                    double h = cost[i, j1] - y[j1] - minD;
                    for (int k = up; k < n; k++)
                    {
                        int j = colList[k];
                        double newDist = cost[i, j] - y[j] - h;
                        if (newDist < d[j])
                        {
                            d[j] = newDist;
                            pred[j] = i;
                            if (newDist == minD)
                            {
                                if (colSol[j] < 0)
                                {
                                    endCol = j;
                                    unassignedFound = true;
                                    break;
                                }

                                colList[k] = colList[up];
                                colList[up++] = j;
                            }
                        }
                    }
                }
            }

            // Update column duals along the scanned set.
            for (int k = 0; k <= last; k++)
            {
                int j1 = colList[k];
                y[j1] += d[j1] - minD;
            }

            // Reconstruct the augmenting path: walk predecessors back to freeRow.
            int col = endCol;
            while (true)
            {
                int i = pred[col];
                colSol[col] = i;
                int tmp = rowSol[i];
                rowSol[i] = col;
                col = tmp;
                if (i == freeRow)
                {
                    break;
                }
            }
        }

        return rowSol;
    }
}

namespace ByteTrack.Matching;

/// <summary>
/// Association helpers. Port of the functions actually used by BYTETracker in
/// <c>yolox/tracker/matching.py</c> (IoU cost, score fusion, linear assignment).
/// ReID-related functions from the original are intentionally omitted.
/// </summary>
public static class Matching
{
    /// <summary>Result of a linear assignment.</summary>
    public readonly record struct AssignmentResult(
        List<(int Track, int Detection)> Matches,
        int[] UnmatchedTracks,
        int[] UnmatchedDetections);

    /// <summary>
    /// IoU overlap matrix between two sets of boxes in <c>tlbr</c>
    /// (x1, y1, x2, y2) format. Replacement for <c>cython_bbox.bbox_overlaps</c>.
    /// </summary>
    public static double[,] Ious(IReadOnlyList<double[]> aTlbrs, IReadOnlyList<double[]> bTlbrs)
    {
        var ious = new double[aTlbrs.Count, bTlbrs.Count];
        if (aTlbrs.Count == 0 || bTlbrs.Count == 0)
        {
            return ious;
        }

        for (int k = 0; k < bTlbrs.Count; k++)
        {
            double[] b = bTlbrs[k];
            double boxArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
            for (int i = 0; i < aTlbrs.Count; i++)
            {
                double[] a = aTlbrs[i];
                double iw = Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]) + 1;
                if (iw <= 0)
                {
                    continue;
                }

                double ih = Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]) + 1;
                if (ih <= 0)
                {
                    continue;
                }

                double ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + boxArea - iw * ih;
                ious[i, k] = iw * ih / ua;
            }
        }

        return ious;
    }

    /// <summary>
    /// IoU-based cost matrix (<c>1 - IoU</c>) between two track lists.
    /// </summary>
    public static double[,] IouDistance(
        IReadOnlyList<STrack> aTracks, IReadOnlyList<STrack> bTracks)
    {
        var aTlbrs = new List<double[]>(aTracks.Count);
        foreach (STrack t in aTracks)
        {
            aTlbrs.Add(t.Tlbr);
        }

        var bTlbrs = new List<double[]>(bTracks.Count);
        foreach (STrack t in bTracks)
        {
            bTlbrs.Add(t.Tlbr);
        }

        double[,] ious = Ious(aTlbrs, bTlbrs);
        int n = ious.GetLength(0);
        int m = ious.GetLength(1);
        var cost = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                cost[i, j] = 1.0 - ious[i, j];
            }
        }

        return cost;
    }

    /// <summary>
    /// Fuses detection confidence into an IoU cost matrix. Port of
    /// <c>fuse_score</c>: <c>fuse_cost = 1 - (1 - cost) * det_score</c>.
    /// </summary>
    public static double[,] FuseScore(double[,] costMatrix, IReadOnlyList<STrack> detections)
    {
        int n = costMatrix.GetLength(0);
        int m = costMatrix.GetLength(1);
        if (n == 0 || m == 0)
        {
            return costMatrix;
        }

        var fused = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                double iouSim = 1.0 - costMatrix[i, j];
                double fuseSim = iouSim * detections[j].Score;
                fused[i, j] = 1.0 - fuseSim;
            }
        }

        return fused;
    }

    /// <summary>
    /// Linear assignment via LAPJV with a cost threshold. Port of
    /// <c>linear_assignment</c>: pairs with cost above <paramref name="thresh"/>
    /// are left unmatched.
    /// </summary>
    public static AssignmentResult LinearAssignment(double[,] costMatrix, double thresh)
    {
        int nRows = costMatrix.GetLength(0);
        int nCols = costMatrix.GetLength(1);

        var matches = new List<(int, int)>();
        if (costMatrix.Length == 0)
        {
            return new AssignmentResult(matches, Range(nRows), Range(nCols));
        }

        (int[] rowToCol, int[] colToRow) = Lapjv.Solve(costMatrix, thresh);

        for (int i = 0; i < nRows; i++)
        {
            if (rowToCol[i] >= 0)
            {
                matches.Add((i, rowToCol[i]));
            }
        }

        var unmatchedA = new List<int>();
        for (int i = 0; i < nRows; i++)
        {
            if (rowToCol[i] < 0)
            {
                unmatchedA.Add(i);
            }
        }

        var unmatchedB = new List<int>();
        for (int j = 0; j < nCols; j++)
        {
            if (colToRow[j] < 0)
            {
                unmatchedB.Add(j);
            }
        }

        return new AssignmentResult(matches, unmatchedA.ToArray(), unmatchedB.ToArray());
    }

    private static int[] Range(int n)
    {
        var r = new int[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = i;
        }

        return r;
    }
}

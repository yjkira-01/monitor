namespace ByteTrack;

/// <summary>
/// A single detection fed into the tracker: a bounding box in <c>tlbr</c>
/// (x1, y1, x2, y2) image coordinates plus a confidence score. The YOLOX
/// detector and its coordinate rescaling are out of scope — supply boxes already
/// in image space.
/// </summary>
public readonly record struct Detection(double X1, double Y1, double X2, double Y2, float Score)
{
    /// <summary>Bounding box as <c>[x1, y1, x2, y2]</c>.</summary>
    public double[] Tlbr => new[] { X1, Y1, X2, Y2 };

    /// <summary>Creates a detection from a tlbr array and a score.</summary>
    public static Detection FromTlbr(double[] tlbr, float score) =>
        new(tlbr[0], tlbr[1], tlbr[2], tlbr[3], score);
}

using ByteTrack.KalmanFilter;

namespace ByteTrack;

/// <summary>
/// A single tracklet. Port of <c>STrack</c> in
/// <c>yolox/tracker/byte_tracker.py</c>.
/// </summary>
public sealed class STrack : BaseTrack
{
    private static readonly KalmanFilter.KalmanFilter SharedKalman = new();

    private readonly double[] _tlwh; // initial detection box, before activation

    private IKalmanFilter? _kalmanFilter;

    public STrack(double[] tlwh, float score)
    {
        _tlwh = (double[])tlwh.Clone();
        Score = score;
        TrackletLen = 0;
        IsActivated = false;
    }

    public double[]? Mean { get; private set; }

    public double[,]? Covariance { get; private set; }

    public int TrackletLen { get; private set; }

    public void Predict()
    {
        double[] meanState = (double[])Mean!.Clone();
        if (State != TrackState.Tracked)
        {
            meanState[7] = 0;
        }

        (Mean, Covariance) = _kalmanFilter!.Predict(meanState, Covariance!);
    }

    /// <summary>Vectorised prediction over many tracks (shared Kalman filter).</summary>
    public static void MultiPredict(IReadOnlyList<STrack> stracks)
    {
        foreach (STrack st in stracks)
        {
            double[] meanState = (double[])st.Mean!.Clone();
            if (st.State != TrackState.Tracked)
            {
                meanState[7] = 0;
            }

            (st.Mean, st.Covariance) = SharedKalman.Predict(meanState, st.Covariance!);
        }
    }

    /// <summary>Starts a new tracklet.</summary>
    public void Activate(IKalmanFilter kalmanFilter, int frameId)
    {
        _kalmanFilter = kalmanFilter;
        TrackId = NextId();
        (Mean, Covariance) = _kalmanFilter.Initiate(TlwhToXyah(_tlwh));

        TrackletLen = 0;
        State = TrackState.Tracked;
        if (frameId == 1)
        {
            IsActivated = true;
        }

        FrameId = frameId;
        StartFrame = frameId;
    }

    /// <summary>Re-activates a lost track using a new detection.</summary>
    public void ReActivate(STrack newTrack, int frameId, bool newId = false)
    {
        (Mean, Covariance) = _kalmanFilter!.Update(Mean!, Covariance!, TlwhToXyah(newTrack.Tlwh));
        TrackletLen = 0;
        State = TrackState.Tracked;
        IsActivated = true;
        FrameId = frameId;
        if (newId)
        {
            TrackId = NextId();
        }

        Score = newTrack.Score;
    }

    /// <summary>Updates a matched track with a new detection.</summary>
    public void Update(STrack newTrack, int frameId)
    {
        FrameId = frameId;
        TrackletLen += 1;

        double[] newTlwh = newTrack.Tlwh;
        (Mean, Covariance) = _kalmanFilter!.Update(Mean!, Covariance!, TlwhToXyah(newTlwh));
        State = TrackState.Tracked;
        IsActivated = true;

        Score = newTrack.Score;
    }

    /// <summary>Current box as <c>(top-left-x, top-left-y, width, height)</c>.</summary>
    public double[] Tlwh
    {
        get
        {
            if (Mean is null)
            {
                return (double[])_tlwh.Clone();
            }

            var ret = new double[4];
            Array.Copy(Mean, ret, 4);
            ret[2] *= ret[3];          // width = aspect * height
            ret[0] -= ret[2] / 2.0;    // top-left x
            ret[1] -= ret[3] / 2.0;    // top-left y
            return ret;
        }
    }

    /// <summary>Current box as <c>(x1, y1, x2, y2)</c>.</summary>
    public double[] Tlbr
    {
        get
        {
            double[] ret = Tlwh;
            ret[2] += ret[0];
            ret[3] += ret[1];
            return ret;
        }
    }

    /// <summary>Converts <c>tlwh</c> to <c>(center-x, center-y, aspect, height)</c>.</summary>
    public static double[] TlwhToXyah(double[] tlwh)
    {
        var ret = (double[])tlwh.Clone();
        ret[0] += ret[2] / 2.0;
        ret[1] += ret[3] / 2.0;
        ret[2] /= ret[3];
        return ret;
    }

    public double[] ToXyah() => TlwhToXyah(Tlwh);

    public static double[] TlbrToTlwh(double[] tlbr)
    {
        var ret = (double[])tlbr.Clone();
        ret[2] -= ret[0];
        ret[3] -= ret[1];
        return ret;
    }

    public static double[] TlwhToTlbr(double[] tlwh)
    {
        var ret = (double[])tlwh.Clone();
        ret[2] += ret[0];
        ret[3] += ret[1];
        return ret;
    }

    public override string ToString() => $"OT_{TrackId}_({StartFrame}-{EndFrame})";
}

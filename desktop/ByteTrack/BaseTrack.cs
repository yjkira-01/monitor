namespace ByteTrack;

/// <summary>
/// Base class for tracklets. Port of <c>BaseTrack</c> in
/// <c>yolox/tracker/basetrack.py</c>.
/// </summary>
/// <remarks>
/// The original keeps a process-global <c>_count</c> on the class. We keep the
/// same static counter for 1:1 behavioural fidelity. Use <see cref="ResetCount"/>
/// between independent runs (e.g. in tests) to get deterministic ids.
/// </remarks>
public abstract class BaseTrack
{
    private static int _count;

    public int TrackId { get; protected set; }

    public bool IsActivated { get; protected set; }

    public TrackState State { get; protected set; } = TrackState.New;

    public float Score { get; protected set; }

    public int StartFrame { get; protected set; }

    public int FrameId { get; protected set; }

    public int TimeSinceUpdate { get; protected set; }

    /// <summary>Frame index at which this track was last updated.</summary>
    public int EndFrame => FrameId;

    /// <summary>Allocates the next globally-unique track id.</summary>
    public static int NextId()
    {
        _count += 1;
        return _count;
    }

    /// <summary>Resets the global id counter. Intended for tests / fresh runs.</summary>
    public static void ResetCount() => _count = 0;

    public void MarkLost() => State = TrackState.Lost;

    public void MarkRemoved() => State = TrackState.Removed;
}

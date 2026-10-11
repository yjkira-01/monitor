namespace ByteTrack;

/// <summary>
/// Lifecycle state of a track. Values mirror the original ByteTrack
/// (<c>yolox/tracker/basetrack.py</c>).
/// </summary>
public enum TrackState
{
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
}

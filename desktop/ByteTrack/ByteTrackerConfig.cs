namespace ByteTrack;

/// <summary>
/// Tracker parameters. Corresponds to the <c>args</c> object consumed by
/// <c>BYTETracker</c> in the original demo (<c>tools/demo_track.py</c>).
/// </summary>
public sealed class ByteTrackerConfig
{
    /// <summary>
    /// High/low detection score split. Detections above this join the first
    /// association; those in <c>(0.1, TrackThresh)</c> join the second.
    /// </summary>
    public float TrackThresh { get; set; } = 0.5f;

    /// <summary>Frames a lost track is retained (at 30 fps) before removal.</summary>
    public int TrackBuffer { get; set; } = 30;

    /// <summary>IoU matching threshold for the first association step.</summary>
    public float MatchThresh { get; set; } = 0.8f;

    /// <summary>
    /// When true (MOT20-style crowded scenes) score fusion is skipped in the
    /// first association and unconfirmed steps.
    /// </summary>
    public bool Mot20 { get; set; }
}

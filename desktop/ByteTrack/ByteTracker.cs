using ByteTrack.KalmanFilter;

namespace ByteTrack;

/// <summary>
/// The ByteTrack multi-object tracker. Port of <c>BYTETracker</c> in
/// <c>yolox/tracker/byte_tracker.py</c>.
/// </summary>
/// <remarks>
/// Feed detections already in image space (the YOLOX detector and its coordinate
/// rescaling are out of scope). Each call to <see cref="Update"/> advances the
/// frame counter and returns the currently active tracks.
/// </remarks>
public sealed class ByteTracker
{
    private const double SecondScoreLow = 0.1;

    private readonly ByteTrackerConfig _args;
    private readonly IKalmanFilter _kalmanFilter;
    private readonly double _detThresh;
    private readonly int _maxTimeLost;

    private List<STrack> _trackedStracks = new();
    private List<STrack> _lostStracks = new();
    private List<STrack> _removedStracks = new();

    /// <summary>
    /// Creates a tracker with the default managed Kalman filter.
    /// </summary>
    /// <param name="args">Tracker parameters.</param>
    /// <param name="frameRate">Source frame rate; scales <see cref="ByteTrackerConfig.TrackBuffer"/>.</param>
    public ByteTracker(ByteTrackerConfig args, int frameRate = 30)
        : this(args, new KalmanFilter.KalmanFilter(), frameRate)
    {
    }

    /// <summary>
    /// Creates a tracker with a custom <see cref="IKalmanFilter"/> implementation.
    /// </summary>
    /// <param name="args">Tracker parameters.</param>
    /// <param name="kalmanFilter">Motion filter used for track prediction.</param>
    /// <param name="frameRate">Source frame rate; scales <see cref="ByteTrackerConfig.TrackBuffer"/>.</param>
    public ByteTracker(ByteTrackerConfig args, IKalmanFilter kalmanFilter, int frameRate = 30)
    {
        _args = args ?? throw new ArgumentNullException(nameof(args));
        _kalmanFilter = kalmanFilter ?? throw new ArgumentNullException(nameof(kalmanFilter));
        FrameId = 0;
        _detThresh = _args.TrackThresh + 0.1;
        int bufferSize = (int)(frameRate / 30.0 * _args.TrackBuffer);
        _maxTimeLost = bufferSize;
    }

    /// <summary>Current frame index (incremented on each <see cref="Update"/>).</summary>
    public int FrameId { get; private set; }

    /// <summary>Currently active (output) tracks after the last update.</summary>
    public IReadOnlyList<STrack> TrackedStracks => _trackedStracks;

    /// <summary>Currently lost tracks.</summary>
    public IReadOnlyList<STrack> LostStracks => _lostStracks;

    /// <summary>
    /// Advances the tracker by one frame and returns the active tracks.
    /// </summary>
    public IReadOnlyList<STrack> Update(IReadOnlyList<Detection> detectionsInput)
    {
        FrameId += 1;
        var activatedStracks = new List<STrack>();
        var refindStracks = new List<STrack>();
        var lostStracks = new List<STrack>();
        var removedStracks = new List<STrack>();

        // Split detections by score into high (remain) and low (second) sets.
        var dets = new List<STrack>();
        var detsSecond = new List<STrack>();
        foreach (Detection d in detectionsInput)
        {
            double[] tlbr = d.Tlbr;
            if (d.Score > _args.TrackThresh)
            {
                dets.Add(new STrack(STrack.TlbrToTlwh(tlbr), d.Score));
            }
            else if (d.Score > SecondScoreLow)
            {
                // inds_low & inds_high: 0.1 < score < track_thresh
                detsSecond.Add(new STrack(STrack.TlbrToTlwh(tlbr), d.Score));
            }
        }

        // Separate confirmed tracked tracks from unconfirmed (single-frame) ones.
        var unconfirmed = new List<STrack>();
        var tracked = new List<STrack>();
        foreach (STrack track in _trackedStracks)
        {
            if (!track.IsActivated)
            {
                unconfirmed.Add(track);
            }
            else
            {
                tracked.Add(track);
            }
        }

        // Step 2: first association with high-score detections.
        List<STrack> strackPool = JointStracks(tracked, _lostStracks);
        STrack.MultiPredict(strackPool);
        double[,] dists = Matching.Matching.IouDistance(strackPool, dets);
        if (!_args.Mot20)
        {
            dists = Matching.Matching.FuseScore(dists, dets);
        }

        Matching.Matching.AssignmentResult a1 =
            Matching.Matching.LinearAssignment(dists, _args.MatchThresh);
        foreach ((int itracked, int idet) in a1.Matches)
        {
            STrack track = strackPool[itracked];
            STrack det = dets[idet];
            if (track.State == TrackState.Tracked)
            {
                track.Update(det, FrameId);
                activatedStracks.Add(track);
            }
            else
            {
                track.ReActivate(det, FrameId, newId: false);
                refindStracks.Add(track);
            }
        }

        // Step 3: second association with low-score detections.
        var rTrackedStracks = new List<STrack>();
        foreach (int i in a1.UnmatchedTracks)
        {
            if (strackPool[i].State == TrackState.Tracked)
            {
                rTrackedStracks.Add(strackPool[i]);
            }
        }

        double[,] dists2 = Matching.Matching.IouDistance(rTrackedStracks, detsSecond);
        Matching.Matching.AssignmentResult a2 =
            Matching.Matching.LinearAssignment(dists2, 0.5);
        foreach ((int itracked, int idet) in a2.Matches)
        {
            STrack track = rTrackedStracks[itracked];
            STrack det = detsSecond[idet];
            if (track.State == TrackState.Tracked)
            {
                track.Update(det, FrameId);
                activatedStracks.Add(track);
            }
            else
            {
                track.ReActivate(det, FrameId, newId: false);
                refindStracks.Add(track);
            }
        }

        foreach (int it in a2.UnmatchedTracks)
        {
            STrack track = rTrackedStracks[it];
            if (track.State != TrackState.Lost)
            {
                track.MarkLost();
                lostStracks.Add(track);
            }
        }

        // Deal with unconfirmed tracks (usually with only one beginning frame).
        var remainingDets = new List<STrack>();
        foreach (int i in a1.UnmatchedDetections)
        {
            remainingDets.Add(dets[i]);
        }

        double[,] dists3 = Matching.Matching.IouDistance(unconfirmed, remainingDets);
        if (!_args.Mot20)
        {
            dists3 = Matching.Matching.FuseScore(dists3, remainingDets);
        }

        Matching.Matching.AssignmentResult a3 =
            Matching.Matching.LinearAssignment(dists3, 0.7);
        foreach ((int itracked, int idet) in a3.Matches)
        {
            unconfirmed[itracked].Update(remainingDets[idet], FrameId);
            activatedStracks.Add(unconfirmed[itracked]);
        }

        foreach (int it in a3.UnmatchedTracks)
        {
            STrack track = unconfirmed[it];
            track.MarkRemoved();
            removedStracks.Add(track);
        }

        // Step 4: init new stracks from still-unmatched high-score detections.
        foreach (int inew in a3.UnmatchedDetections)
        {
            STrack track = remainingDets[inew];
            if (track.Score < _detThresh)
            {
                continue;
            }

            track.Activate(_kalmanFilter, FrameId);
            activatedStracks.Add(track);
        }

        // Step 5: update state — remove long-lost tracks.
        foreach (STrack track in _lostStracks)
        {
            if (FrameId - track.EndFrame > _maxTimeLost)
            {
                track.MarkRemoved();
                removedStracks.Add(track);
            }
        }

        // Merge track lists.
        _trackedStracks = _trackedStracks.Where(t => t.State == TrackState.Tracked).ToList();
        _trackedStracks = JointStracks(_trackedStracks, activatedStracks);
        _trackedStracks = JointStracks(_trackedStracks, refindStracks);
        _lostStracks = SubStracks(_lostStracks, _trackedStracks);
        _lostStracks.AddRange(lostStracks);
        _lostStracks = SubStracks(_lostStracks, _removedStracks);
        _removedStracks.AddRange(removedStracks);
        // CareMonitor: ids are never reused; older removed tracks cannot reappear.
        // Retain enough recent removals for next-update subtraction without growing forever.
        if (_removedStracks.Count > 256) _removedStracks.RemoveRange(0, _removedStracks.Count - 256);
        (_trackedStracks, _lostStracks) = RemoveDuplicateStracks(_trackedStracks, _lostStracks);

        return _trackedStracks.Where(t => t.IsActivated).ToList();
    }

    internal static List<STrack> JointStracks(
        IReadOnlyList<STrack> tlistA, IReadOnlyList<STrack> tlistB)
    {
        var exists = new HashSet<int>();
        var res = new List<STrack>();
        foreach (STrack t in tlistA)
        {
            exists.Add(t.TrackId);
            res.Add(t);
        }

        foreach (STrack t in tlistB)
        {
            if (exists.Add(t.TrackId))
            {
                res.Add(t);
            }
        }

        return res;
    }

    internal static List<STrack> SubStracks(
        IReadOnlyList<STrack> tlistA, IReadOnlyList<STrack> tlistB)
    {
        var stracks = new Dictionary<int, STrack>();
        foreach (STrack t in tlistA)
        {
            stracks[t.TrackId] = t;
        }

        foreach (STrack t in tlistB)
        {
            stracks.Remove(t.TrackId);
        }

        return stracks.Values.ToList();
    }

    internal static (List<STrack> ResA, List<STrack> ResB) RemoveDuplicateStracks(
        IReadOnlyList<STrack> stracksA, IReadOnlyList<STrack> stracksB)
    {
        double[,] pdist = Matching.Matching.IouDistance(stracksA, stracksB);
        var dupA = new HashSet<int>();
        var dupB = new HashSet<int>();
        int n = pdist.GetLength(0);
        int m = pdist.GetLength(1);
        for (int p = 0; p < n; p++)
        {
            for (int q = 0; q < m; q++)
            {
                if (pdist[p, q] < 0.15)
                {
                    int timeP = stracksA[p].FrameId - stracksA[p].StartFrame;
                    int timeQ = stracksB[q].FrameId - stracksB[q].StartFrame;
                    if (timeP > timeQ)
                    {
                        dupB.Add(q);
                    }
                    else
                    {
                        dupA.Add(p);
                    }
                }
            }
        }

        var resA = new List<STrack>();
        for (int i = 0; i < stracksA.Count; i++)
        {
            if (!dupA.Contains(i))
            {
                resA.Add(stracksA[i]);
            }
        }

        var resB = new List<STrack>();
        for (int i = 0; i < stracksB.Count; i++)
        {
            if (!dupB.Contains(i))
            {
                resB.Add(stracksB[i]);
            }
        }

        return (resA, resB);
    }
}

namespace ByteTrack.KalmanFilter;

/// <summary>
/// Constant-velocity Kalman filter over the 8-D bounding-box state
/// <c>(x, y, a, h, vx, vy, va, vh)</c>. Port of
/// <c>yolox/tracker/kalman_filter.py</c>. Abstracted so an alternative
/// implementation can be substituted.
/// </summary>
public interface IKalmanFilter
{
    /// <summary>Creates a track from an unassociated measurement <c>(x, y, a, h)</c>.</summary>
    (double[] Mean, double[,] Covariance) Initiate(double[] measurement);

    /// <summary>Prediction step (single track).</summary>
    (double[] Mean, double[,] Covariance) Predict(double[] mean, double[,] covariance);

    /// <summary>Projects the state distribution into measurement space.</summary>
    (double[] Mean, double[,] Covariance) Project(double[] mean, double[,] covariance);

    /// <summary>Correction step against a measurement <c>(x, y, a, h)</c>.</summary>
    (double[] Mean, double[,] Covariance) Update(
        double[] mean, double[,] covariance, double[] measurement);
}

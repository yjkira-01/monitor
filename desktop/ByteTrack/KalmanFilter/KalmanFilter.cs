using ByteTrack.LinearAlgebra;

namespace ByteTrack.KalmanFilter;

/// <summary>
/// A simple Kalman filter for tracking bounding boxes in image space.
/// Port of <c>KalmanFilter</c> in <c>yolox/tracker/kalman_filter.py</c>.
/// </summary>
/// <remarks>
/// The 8-dimensional state space is <c>(x, y, a, h, vx, vy, va, vh)</c>: the
/// bounding-box center position (x, y), aspect ratio a, height h and their
/// velocities. Motion follows a constant-velocity model; the box (x, y, a, h)
/// is a direct linear observation of the state.
/// </remarks>
public sealed class KalmanFilter : IKalmanFilter
{
    /// <summary>
    /// 0.95 quantile of the chi-square distribution with N degrees of freedom
    /// (N = 1..9), used as the Mahalanobis gating threshold.
    /// </summary>
    public static readonly IReadOnlyDictionary<int, double> Chi2Inv95 =
        new Dictionary<int, double>
        {
            [1] = 3.8415,
            [2] = 5.9915,
            [3] = 7.8147,
            [4] = 9.4877,
            [5] = 11.070,
            [6] = 12.592,
            [7] = 14.067,
            [8] = 15.507,
            [9] = 16.919,
        };

    private const int Ndim = 4;

    private readonly ILinearAlgebra _la;
    private readonly double[,] _motionMat;   // 8×8
    private readonly double[,] _updateMat;   // 4×8
    private readonly double[,] _motionMatT;  // 8×8 (cached transpose)
    private readonly double[,] _updateMatT;  // 8×4 (cached transpose)
    private readonly double _stdWeightPosition = 1.0 / 20.0;
    private readonly double _stdWeightVelocity = 1.0 / 160.0;

    public KalmanFilter()
        : this(ManagedLinearAlgebra.Instance)
    {
    }

    public KalmanFilter(ILinearAlgebra linearAlgebra)
    {
        _la = linearAlgebra ?? throw new ArgumentNullException(nameof(linearAlgebra));

        const double dt = 1.0;
        _motionMat = Identity(2 * Ndim);
        for (int i = 0; i < Ndim; i++)
        {
            _motionMat[i, Ndim + i] = dt;
        }

        _updateMat = new double[Ndim, 2 * Ndim];
        for (int i = 0; i < Ndim; i++)
        {
            _updateMat[i, i] = 1.0;
        }

        _motionMatT = _la.Transpose(_motionMat);
        _updateMatT = _la.Transpose(_updateMat);
    }

    public (double[] Mean, double[,] Covariance) Initiate(double[] measurement)
    {
        var mean = new double[2 * Ndim];
        Array.Copy(measurement, mean, Ndim); // velocities initialised to 0

        double h = measurement[3];
        double[] std =
        {
            2 * _stdWeightPosition * h,
            2 * _stdWeightPosition * h,
            1e-2,
            2 * _stdWeightPosition * h,
            10 * _stdWeightVelocity * h,
            10 * _stdWeightVelocity * h,
            1e-5,
            10 * _stdWeightVelocity * h,
        };

        return (mean, DiagSquare(std));
    }

    public (double[] Mean, double[,] Covariance) Predict(double[] mean, double[,] covariance)
    {
        double h = mean[3];
        double[] std =
        {
            _stdWeightPosition * h,
            _stdWeightPosition * h,
            1e-2,
            _stdWeightPosition * h,
            _stdWeightVelocity * h,
            _stdWeightVelocity * h,
            1e-5,
            _stdWeightVelocity * h,
        };
        double[,] motionCov = DiagSquare(std);

        // mean = motion_mat · mean
        double[] newMean = _la.MatVec(_motionMat, mean);

        // covariance = motion_mat · covariance · motion_matᵀ + motion_cov
        double[,] newCov = _la.MatMul(_la.MatMul(_motionMat, covariance), _motionMatT);
        newCov = _la.Add(newCov, motionCov);

        return (newMean, newCov);
    }

    public (double[] Mean, double[,] Covariance) Project(double[] mean, double[,] covariance)
    {
        double h = mean[3];
        double[] std =
        {
            _stdWeightPosition * h,
            _stdWeightPosition * h,
            1e-1,
            _stdWeightPosition * h,
        };
        double[,] innovationCov = DiagSquare(std);

        // mean = update_mat · mean
        double[] projMean = _la.MatVec(_updateMat, mean);

        // covariance = update_mat · covariance · update_matᵀ + innovation_cov
        double[,] projCov = _la.MatMul(_la.MatMul(_updateMat, covariance), _updateMatT);
        projCov = _la.Add(projCov, innovationCov);

        return (projMean, projCov);
    }

    public (double[] Mean, double[,] Covariance) Update(
        double[] mean, double[,] covariance, double[] measurement)
    {
        var (projMean, projCov) = Project(mean, covariance);

        // Original:
        //   chol = cho_factor(proj_cov, lower=True)
        //   kalman_gain = cho_solve(chol, (covariance · update_matᵀ).T).T
        // Let B = covariance · update_matᵀ  (8×4).
        // cho_solve solves  proj_cov · Z = Bᵀ   for Z (4×8),
        // and kalman_gain = Zᵀ  (8×4).
        double[,] cholLower = _la.CholeskyLower(projCov);
        double[,] b = _la.MatMul(covariance, _updateMatT);      // 8×4
        double[,] bT = _la.Transpose(b);                        // 4×8
        double[,] z = _la.CholeskySolve(cholLower, bT);         // 4×8
        double[,] kalmanGain = _la.Transpose(z);                // 8×4

        // innovation = measurement - proj_mean  (4)
        var innovation = new double[Ndim];
        for (int i = 0; i < Ndim; i++)
        {
            innovation[i] = measurement[i] - projMean[i];
        }

        // new_mean = mean + kalman_gain · innovation  (8)
        double[] gainInnovation = _la.MatVec(kalmanGain, innovation);
        var newMean = new double[2 * Ndim];
        for (int i = 0; i < 2 * Ndim; i++)
        {
            newMean[i] = mean[i] + gainInnovation[i];
        }

        // new_covariance = covariance - kalman_gain · proj_cov · kalman_gainᵀ
        double[,] kalmanGainT = _la.Transpose(kalmanGain);      // 4×8
        double[,] tmp = _la.MatMul(_la.MatMul(kalmanGain, projCov), kalmanGainT); // 8×8
        var newCov = new double[2 * Ndim, 2 * Ndim];
        for (int i = 0; i < 2 * Ndim; i++)
        {
            for (int j = 0; j < 2 * Ndim; j++)
            {
                newCov[i, j] = covariance[i, j] - tmp[i, j];
            }
        }

        return (newMean, newCov);
    }

    /// <summary>
    /// Squared Mahalanobis gating distance between the (projected) state and a
    /// set of measurements. Port of <c>gating_distance</c> with
    /// <c>metric='maha'</c>, <c>only_position=false</c>.
    /// </summary>
    public double[] GatingDistance(double[] mean, double[,] covariance, double[][] measurements)
    {
        var (projMean, projCov) = Project(mean, covariance);

        // d = measurements - proj_mean  →  build dᵀ (4 × N) as RHS.
        int n = measurements.Length;
        var dT = new double[Ndim, n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < Ndim; i++)
            {
                dT[i, j] = measurements[j][i] - projMean[i];
            }
        }

        double[,] cholesky = _la.CholeskyLower(projCov);
        // z = L⁻¹ · dᵀ  (solve L · z = dᵀ, lower triangular)
        double[,] z = _la.SolveTriangular(cholesky, dT, lower: true);

        var result = new double[n];
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < Ndim; i++)
            {
                sum += z[i, j] * z[i, j];
            }

            result[j] = sum;
        }

        return result;
    }

    private static double[,] Identity(int n)
    {
        var m = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            m[i, i] = 1.0;
        }

        return m;
    }

    private static double[,] DiagSquare(double[] std)
    {
        int n = std.Length;
        var m = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            m[i, i] = std[i] * std[i];
        }

        return m;
    }
}

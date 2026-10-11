namespace ByteTrack.LinearAlgebra;

/// <summary>
/// Dense linear-algebra operations required by the tracker's Kalman filter.
/// Abstracted behind an interface so the default managed implementation can be
/// swapped for a faster/native one (e.g. MathNet.Numerics) without touching the
/// filter code.
/// </summary>
/// <remarks>
/// Matrices are row-major <c>double[rows, cols]</c>; vectors are <c>double[]</c>.
/// </remarks>
public interface ILinearAlgebra
{
    /// <summary>Matrix product <c>a · b</c>.</summary>
    double[,] MatMul(double[,] a, double[,] b);

    /// <summary>Matrix–vector product <c>a · v</c>.</summary>
    double[] MatVec(double[,] a, double[] v);

    /// <summary>Vector–matrix product <c>v · a</c> (v treated as a row vector).</summary>
    double[] VecMat(double[] v, double[,] a);

    /// <summary>Transpose of <paramref name="a"/>.</summary>
    double[,] Transpose(double[,] a);

    /// <summary>Element-wise sum <c>a + b</c>.</summary>
    double[,] Add(double[,] a, double[,] b);

    /// <summary>Lower-triangular Cholesky factor L such that <c>a = L · Lᵀ</c>.</summary>
    double[,] CholeskyLower(double[,] a);

    /// <summary>
    /// Solves <c>a · X = b</c> given the lower Cholesky factor of the SPD matrix
    /// <c>a</c>. Right-hand side <paramref name="b"/> has one column per system.
    /// Equivalent to <c>scipy.linalg.cho_solve((L, lower=True), b)</c>.
    /// </summary>
    double[,] CholeskySolve(double[,] choleskyLower, double[,] b);

    /// <summary>
    /// Solves the triangular system <c>L · x = b</c> (or <c>Lᵀ · x = b</c> when
    /// <paramref name="lower"/> is false) for each column of <paramref name="b"/>.
    /// Equivalent to <c>scipy.linalg.solve_triangular</c>.
    /// </summary>
    double[,] SolveTriangular(double[,] l, double[,] b, bool lower);
}

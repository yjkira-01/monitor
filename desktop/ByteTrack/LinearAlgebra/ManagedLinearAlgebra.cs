namespace ByteTrack.LinearAlgebra;

/// <summary>
/// Pure-managed <see cref="ILinearAlgebra"/> implementation with no external
/// dependencies. Uses textbook algorithms; the matrices involved in tracking are
/// small (≤ 8×8), so this is more than fast enough.
/// </summary>
public sealed class ManagedLinearAlgebra : ILinearAlgebra
{
    /// <summary>Shared stateless singleton.</summary>
    public static readonly ManagedLinearAlgebra Instance = new();

    public double[,] MatMul(double[,] a, double[,] b)
    {
        int n = a.GetLength(0);
        int m = a.GetLength(1);
        int p = b.GetLength(1);
        if (m != b.GetLength(0))
        {
            throw new ArgumentException("Inner matrix dimensions must agree.");
        }

        var result = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < m; k++)
            {
                double aik = a[i, k];
                if (aik == 0.0)
                {
                    continue;
                }

                for (int j = 0; j < p; j++)
                {
                    result[i, j] += aik * b[k, j];
                }
            }
        }

        return result;
    }

    public double[] MatVec(double[,] a, double[] v)
    {
        int n = a.GetLength(0);
        int m = a.GetLength(1);
        if (m != v.Length)
        {
            throw new ArgumentException("Matrix columns must match vector length.");
        }

        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < m; j++)
            {
                sum += a[i, j] * v[j];
            }

            result[i] = sum;
        }

        return result;
    }

    public double[] VecMat(double[] v, double[,] a)
    {
        int n = a.GetLength(0);
        int m = a.GetLength(1);
        if (n != v.Length)
        {
            throw new ArgumentException("Vector length must match matrix rows.");
        }

        var result = new double[m];
        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                sum += v[i] * a[i, j];
            }

            result[j] = sum;
        }

        return result;
    }

    public double[,] Transpose(double[,] a)
    {
        int n = a.GetLength(0);
        int m = a.GetLength(1);
        var result = new double[m, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                result[j, i] = a[i, j];
            }
        }

        return result;
    }

    public double[,] Add(double[,] a, double[,] b)
    {
        int n = a.GetLength(0);
        int m = a.GetLength(1);
        if (n != b.GetLength(0) || m != b.GetLength(1))
        {
            throw new ArgumentException("Matrix dimensions must agree.");
        }

        var result = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                result[i, j] = a[i, j] + b[i, j];
            }
        }

        return result;
    }

    public double[,] CholeskyLower(double[,] a)
    {
        int n = a.GetLength(0);
        if (n != a.GetLength(1))
        {
            throw new ArgumentException("Cholesky requires a square matrix.");
        }

        var l = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = a[i, j];
                for (int k = 0; k < j; k++)
                {
                    sum -= l[i, k] * l[j, k];
                }

                if (i == j)
                {
                    if (sum <= 0.0)
                    {
                        throw new ArgumentException(
                            "Matrix is not positive definite.");
                    }

                    l[i, j] = Math.Sqrt(sum);
                }
                else
                {
                    l[i, j] = sum / l[j, j];
                }
            }
        }

        return l;
    }

    public double[,] CholeskySolve(double[,] choleskyLower, double[,] b)
    {
        // a = L Lᵀ  ⇒  solve L y = b (forward), then Lᵀ x = y (backward).
        double[,] y = ForwardSubstitution(choleskyLower, b);
        return BackSubstitutionTransposed(choleskyLower, y);
    }

    public double[,] SolveTriangular(double[,] l, double[,] b, bool lower)
    {
        return lower
            ? ForwardSubstitution(l, b)
            : BackSubstitution(l, b);
    }

    /// <summary>Solves <c>L · X = b</c> for lower-triangular L.</summary>
    private static double[,] ForwardSubstitution(double[,] l, double[,] b)
    {
        int n = l.GetLength(0);
        int cols = b.GetLength(1);
        var x = new double[n, cols];
        for (int c = 0; c < cols; c++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = b[i, c];
                for (int k = 0; k < i; k++)
                {
                    sum -= l[i, k] * x[k, c];
                }

                x[i, c] = sum / l[i, i];
            }
        }

        return x;
    }

    /// <summary>Solves <c>U · X = b</c> for upper-triangular U.</summary>
    private static double[,] BackSubstitution(double[,] u, double[,] b)
    {
        int n = u.GetLength(0);
        int cols = b.GetLength(1);
        var x = new double[n, cols];
        for (int c = 0; c < cols; c++)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = b[i, c];
                for (int k = i + 1; k < n; k++)
                {
                    sum -= u[i, k] * x[k, c];
                }

                x[i, c] = sum / u[i, i];
            }
        }

        return x;
    }

    /// <summary>Solves <c>Lᵀ · X = b</c> given lower-triangular L.</summary>
    private static double[,] BackSubstitutionTransposed(double[,] l, double[,] b)
    {
        int n = l.GetLength(0);
        int cols = b.GetLength(1);
        var x = new double[n, cols];
        for (int c = 0; c < cols; c++)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = b[i, c];
                for (int k = i + 1; k < n; k++)
                {
                    // (Lᵀ)[i,k] = L[k,i]
                    sum -= l[k, i] * x[k, c];
                }

                x[i, c] = sum / l[i, i];
            }
        }

        return x;
    }
}

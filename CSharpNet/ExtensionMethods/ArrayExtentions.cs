using System;

namespace SimpleDeepNet.ExtensionMethods;

public static class ArrayExtentions
{
    public static double[,] Multiply(this double[,] matrix1, double[,] matrix2)
    {
        var matrix1Rows = matrix1.GetLength(0);
        var matrix1Cols = matrix1.GetLength(1);
        var matrix2Rows = matrix2.GetLength(0);
        var matrix2Cols = matrix2.GetLength(1);

        if (matrix1Cols != matrix2Rows)
        {
            throw new Exception("the Number Of First Matrix Columns Should Be Equal With Number Of Rows In Second Matrix");
        }

        var result = new double[matrix1Rows, matrix2Cols];

        for (int i = 0; i < matrix1Rows; i++)
        {
            for (int j = 0; j < matrix2Cols; j++)
            {
                double sum = 0.0;

                for (int k = 0; k < matrix1Cols; k++)
                {
                    sum += matrix1[i, k] * matrix2[k, j];
                }

                result[i, j] = sum;
            }
        }

        return result;
    }

    public static double[] Multiply(this double[] array1, double[,] matrix2)
    {
        var array1Rows = 1;
        var array1Cols = array1.Length;
        var matrix2Rows = matrix2.GetLength(0);
        var matrix2Cols = matrix2.GetLength(1);

        if (array1Cols != matrix2Rows)
        {
            throw new Exception("the Number Of First Matrix Columns Should Be Equal With Number Of Rows In Second Matrix");
        }

        var result = new double[matrix2Cols];

        for (int i = 0; i < array1Rows; i++)
        {
            for (int j = 0; j < matrix2Cols; j++)
            {
                double sum = 0.0;

                for (int k = 0; k < array1Cols; k++)
                {
                    sum += array1[k] * matrix2[k, j];
                }

                result[j] = sum;
            }
        }

        return result;
    }

    public static double[] Subtract(this double[] array1, double[] array2)
    {
        if (array1.Length != array2.Length)
        {
            throw new Exception("Two Arrays Should Have Equal Lenght");
        }

        var result = new double[array1.Length];

        for (int i = 0; i < array1.Length; i++)
        {
            result[i] = array1[i] - array2[i];
        }

        return result;
    }

    public static double[,] Transpose(this double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);

        var result = new double[cols, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int ii = 0; ii < cols; ii++)
            {
                result[ii, i] = matrix[i, ii];
            }
        }

        return result;
    }

    public static double[,] ConvertTo2DMatrix(this double[] array)
    {
        var rows = 1;
        var cols = array.Length;

        var result = new double[rows, cols];

        for (int i = 0; i < cols; i++)
        {
            result[0, i] = array[i];
        }

        return result;
    }
}
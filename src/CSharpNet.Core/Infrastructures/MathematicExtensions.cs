using System;
using System.Collections.Generic;

namespace CSharpNet.Infrastructures;

public static class MathematicExtensions
{
    /// <summary>
    /// Multiply 2 Dimensional Matrices
    /// </summary>
    /// <param name="matrix1"></param>
    /// <param name="matrix2"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
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

    public static int[,] Mul(this float[,] matrix1, int[,] matrix2)
    {
        var matrix1Rows = matrix1.GetLength(0);
        var matrix1Columns = matrix1.GetLength(1);

        var matrix2Rows = matrix2.GetLength(0);
        var matrix2Columns = matrix2.GetLength(1);

        if (matrix1Rows != matrix2Rows && matrix1Columns != matrix2Columns)
            throw new Exception("Columns And Rows In Both Matrixes Should Be Equal.");

        var result = new int[matrix1Rows, matrix1Columns];

        for (int i = 0; i < matrix1Rows; i++)
        {
            for (int ii = 0; ii < matrix1Columns; ii++)
            {
                result[i, ii] = (int)(matrix1[i, ii] * matrix2[i, ii]);
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

    public static List<double[][]> SerializeToJsonArray(this List<double[,]> arrays)
    {
        var outputList = new List<double[][]>();
        for (int arrayIndex = 0; arrayIndex < arrays.Count; arrayIndex++)
        {
            var rows = arrays[arrayIndex].GetLength(0);
            var cols = arrays[arrayIndex].GetLength(1);

            var result = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                result[i] = new double[cols];
                for (int ii = 0; ii < cols; ii++)
                {
                    result[i][ii] = arrays[arrayIndex][i, ii];
                }
            }

            outputList.Add(result);
        }


        return outputList;
    }

    public static List<double[,]> DeserializeTo2DArray(this List<double[][]> arrays)
    {
        var outputList = new List<double[,]>();
        for (int arrayIndex = 0; arrayIndex < arrays.Count; arrayIndex++)
        {
            var rows = arrays[arrayIndex].Length;
            var cols = arrays[arrayIndex][0].Length;

            var result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int ii = 0; ii < cols; ii++)
                {
                    result[i, ii] = arrays[arrayIndex][i][ii];
                }
            }

            outputList.Add(result);
        }


        return outputList;
    }

    public static double[] Sumation(this double[] array, double collector)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = array[i] + collector;
        }

        return array;
    }

    public static int Sum(this int[,] matrix1)
    {
        var matrix1Rows = matrix1.GetLength(0);
        var matrix1Columns = matrix1.GetLength(1);

        var result = 0;

        for (int i = 0; i < matrix1Rows; i++)
        {
            for (int ii = 0; ii < matrix1Columns; ii++)
            {
                result += matrix1[i, ii];
            }
        }

        return result;
    }

    public static int ToInt(this string text)
    {
        var output = 0;
        try
        {
            output = int.Parse(text);
        }
        catch (Exception)
        {
            throw;
        }

        return output;
    }

    public static float ToFloat(this string text)
    {
        var output = 0.0f;
        try
        {
            output = float.Parse(text);
        }
        catch (Exception)
        {
            throw;
        }

        return output;
    }
}
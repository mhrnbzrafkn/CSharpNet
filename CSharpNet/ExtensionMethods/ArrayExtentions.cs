using System;

namespace CSharpNet.FeedForwardNet.ExtensionMethods;

public static class ArrayExtentions
{
    public static double[,] Mul(this double[,] arr1, double[,] arr2)
    {
        int rows1 = arr1.GetLength(0);
        int cols1 = arr1.GetLength(1);
        int rows2 = arr2.GetLength(0);
        int cols2 = arr2.GetLength(1);

        if (cols1 != rows2)
        {
            throw new ArgumentException("The number of columns in the first array must match the number of rows in the second array.");
        }

        double[,] result = new double[rows1, cols2];

        for (int i = 0; i < rows1; i++)
        {
            for (int j = 0; j < cols2; j++)
            {
                double sum = 0;
                for (int k = 0; k < cols1; k++)
                {
                    sum += arr1[i, k] * arr2[k, j];
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    public static double[] Mul(this double[] array1, double[][] array2)
    {
        int rows1 = 1;
        int cols1 = array1.Length;
        int rows2 = array2.Length;
        int cols2 = array2[0].Length;

        if (cols1 != rows2)
        {
            throw new ArgumentException("The number of columns in the first array must match the number of rows in the second array.");
        }

        double[] result = new double[rows1 * cols2];
        for (int i = 0; i < rows1; i++)
        {
            for (int j = 0; j < cols2; j++)
            {
                double sum = 0;
                for (int k = 0; k < cols1; k++)
                {
                    sum += array1[k] * array2[k][j];
                }
                result[i + j] = sum;
            }
        }

        return result;
    }

    public static double[] Mul(this double[] array1, double[] array2)
    {
        var newArray = new double[array1.Length];
        for (int i = 0; i < array1.Length; i++)
        {
            newArray[i] = array1[i] * array2[i];
        }

        return newArray;
    }

    public static double[] Sub(this double[] array1, double[] array2)
    {
        var errors = new double[array1.Length];
        for (int i = 0; i < array1.Length; i++)
        {
            errors[i] = array1[i] - array2[i];
        }

        return errors;
    }

    public static double[] Sub(this int num, double[] array)
    {
        var newValues = new double[array.Length];
        for (int i = 0; i < array.Length; i++)
        {
            newValues[i] = num - array[i];
        }

        return newValues;
    }

    public static double[][] Dot(this double num, double[][] array)
    {
        var newValues = new double[array.Length][];
        var rows = array.Length;
        var cols = array[0].Length;
        for (int i = 0; i < rows; i++)
        {
            newValues[i] = new double[cols];
            for (int ii = 0; ii < cols; ii++)
            {
                newValues[i][ii] = num * array[i][ii];
            }
        }

        return newValues;
    }

    public static double[] Dot(this double num, double[] array)
    {
        var newValues = new double[array.Length];
        for (int i = 0; i < array.Length; i++)
        {
            newValues[i] = num * array[i];
        }

        return newValues;
    }

    public static double[][] Sum(this double[][] array1, double[][] array2)
    {
        var newArray = new double[array1.Length][];
        var rows = array1.Length;
        var cols = array2[0].Length;

        for (int i = 0; i < rows; i++)
        {
            newArray[i] = new double[cols];
            for (int ii = 0; ii < cols; ii++)
            {
                newArray[i][ii] = array1[i][ii] + array2[i][ii];
            }
        }

        return newArray;
    }

    public static double[][] T(this double[][] array)
    {
        var rows = array.Length;
        var cols = array[0].Length;
        var newArray = new double[cols][];
        for (int i = 0; i < cols; i++)
        {
            newArray[i] = new double[rows];
            for (int ii = 0; ii < rows; ii++)
            {
                newArray[i][ii] = array[ii][i];
            }
        }

        return newArray;
    }

    public static double[][] outer(this double[] array1, double[] array2)
    {
        var rows = array1.Length;
        var cols = array2.Length;
        var outer = new double[array1.Length][];
        for (int i = 0; i < rows; i++)
        {
            outer[i] = new double[cols];
            for (int ii = 0; ii < cols; ii++)
            {
                outer[i][ii] = array1[i] * array2[ii];
            }
        }

        return outer;
    }

    public static double[] ConvertToFlat(this double[,] array)
    {
        var rows = array.GetLength(0);
        var cols = array.GetLength(1);
        var flatArray = new double[rows * cols];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                var flatIndex = row * cols + col;
                flatArray[flatIndex] = array[row, col];
            }
        }

        return flatArray;
    }
}
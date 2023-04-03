using System;

namespace CSharpNet;

public class Matrix
{
    public static double[,] Generate(int rows, int cols)
    {
        var matrix = new double[rows, cols];
        var rnd = new Random();

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                matrix[row, col] = (double)rnd.Next(-10000, 10000) / 10000;
            }
        }

        return matrix;
    }
}

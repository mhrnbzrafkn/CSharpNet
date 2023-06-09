﻿using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace CSharpNet;

/// <summary>
/// Loading Image DataSets
/// </summary>
public class DataSetLoader
{
    public List<double[,]> Data { get; set; }
    public List<double[]> ActualOutputs { get; set; }
    public List<double[,]> TestData { get; set; }

    public DataSetLoader()
    {
        Data = new List<double[,]>();
        ActualOutputs = new List<double[]>();
        TestData = new List<double[,]>();
    }

    private void AddNewData(double[,] data, double[] actualOutput)
    {
        Data.Add(data);
        ActualOutputs.Add(actualOutput);
    }

    public void LoadDataFromFolder(string dataPath, int imageWidth, int imageHeight)
    {
        var positiveFilesPath = dataPath + "\\p";
        var positiveFiles = Directory.GetFiles(positiveFilesPath);
        foreach (var positiveImagePath in positiveFiles)
        {
            var image = LoadImage(positiveImagePath, imageWidth, imageHeight);
            var doubleImage = ConvertToDoubleMatrix(image);
            var oneDImage = ConvertToOneDMatxi(doubleImage);
            var normalizedImage = NormalizeData(oneDImage);
            AddNewData(normalizedImage, new double[] { 1, -1 });
        }

        var negativeFilesPath = dataPath + "\\n";
        var negativeFiles = Directory.GetFiles(negativeFilesPath);
        foreach (var negativeImagePath in negativeFiles)
        {
            var image = LoadImage(negativeImagePath, imageWidth, imageHeight);
            var doubleImage = ConvertToDoubleMatrix(image);
            var oneDImage = ConvertToOneDMatxi(doubleImage);
            var normalizedImage = NormalizeData(oneDImage);
            AddNewData(normalizedImage, new double[] { -1, 1 });
        }

        var testFilesPath = dataPath + "\\test";
        var testFiles = Directory.GetFiles(testFilesPath);
        foreach (var testImagePath in testFiles)
        {
            var image = LoadImage(testImagePath, imageWidth, imageHeight);
            var doubleImage = ConvertToDoubleMatrix(image);
            var oneDImage = ConvertToOneDMatxi(doubleImage);
            var normalizedImage = NormalizeData(oneDImage);
            TestData.Add(normalizedImage);
        }
    }

    private double[,] NormalizeData(double[,] data)
    {
        int cols = data.GetLength(1);
        double[,] normalizedData = new double[1, cols];

        for (int index = 0; index < data.GetLength(1); index++)
        {
            var value = data[0, index];
            normalizedData[0, index] = value / 255;
        }

        return normalizedData;
    }

    private double[,] ConvertToOneDMatxi(double[,] image)
    {
        int rows = image.GetLength(0);
        int cols = image.GetLength(1);

        double[,] doubleArray = new double[1, rows * cols];

        for (int rowIndex = 0; rowIndex < rows; rowIndex++)
        {
            for (int colIndex = 0; colIndex < cols; colIndex++)
            {
                var index = (rowIndex * cols) + colIndex;
                doubleArray[0, index] = image[rowIndex, colIndex];
            }
        }

        return doubleArray;
    }

    private int[,] LoadImage(string imagePath, int imageWidth, int imageHeight)
    {
        var originalImage = new Bitmap(imagePath);
        var bitmap = new Bitmap(originalImage, imageWidth, imageHeight);

        // Create a matrix to hold the pixel data
        int[,] matrix = new int[bitmap.Width, bitmap.Height];

        // Loop through each pixel and store the RGB value in the matrix
        for (int x = 0; x < bitmap.Width; x++)
        {
            for (int y = 0; y < bitmap.Height; y++)
            {
                Color pixel = bitmap.GetPixel(x, y);
                int red = pixel.R;
                int green = pixel.G;
                int blue = pixel.B;
                var grayColor = (red + green + blue) / 3;
                matrix[x, y] = grayColor;
            }
        }

        return matrix;
    }

    private double[,] ConvertToDoubleMatrix(int[,] intMatrix)
    {
        int rows = intMatrix.GetLength(0);
        int cols = intMatrix.GetLength(1);

        double[,] doubleArray = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                doubleArray[i, j] = intMatrix[i, j];
            }
        }

        return doubleArray;
    }
}
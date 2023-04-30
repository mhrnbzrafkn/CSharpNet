using CSharpNet.Infrastructures;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace CSharpNet;

/// <summary>
/// Process Images, Prepare For Feeding To Neural Network
/// </summary>
public class ImageProcessor
{
    /// <summary>
    /// Create Convolution With Specific Kernel
    /// </summary>
    /// <param name="matrix"></param>
    /// <param name="kernelWidth"></param>
    /// <param name="kernelHeight"></param>
    /// <returns></returns>
    public int[,] CreateConvolution(
        int[,] matrix,
        int kernelWidth,
        int kernelHeight)
    {
        var width = matrix.GetLength(0);
        var height = matrix.GetLength(1);

        var biggerMatrixWidthEdge = kernelWidth - 1;
        var biggerMatrixHeightEdge = kernelWidth - 1;

        var biggerMatrix = new int[
            width + biggerMatrixWidthEdge,
            height + biggerMatrixHeightEdge];

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                biggerMatrix[
                    x + (biggerMatrixWidthEdge / 2),
                    y + (biggerMatrixHeightEdge / 2)] = matrix[x, y];
            }
        }

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                var sum = 0.0;
                for (int i = 0; i < kernelWidth; i++)
                {
                    for (int j = 0; j < kernelHeight; j++)
                    {
                        sum += biggerMatrix[x + i, y + j];
                    }
                }
                var result = sum / (kernelWidth * kernelHeight);

                matrix[x, y] = (int)result;
            }
        }

        return matrix;
    }

    /// <summary>
    /// Create Polling Max With Specific Kernel Size
    /// </summary>
    /// <param name="matrix"></param>
    /// <param name="kernelWidth"></param>
    /// <param name="kernelHeight"></param>
    /// <returns></returns>
    public int[,] CreatePollingMax(
        int[,] matrix,
        int kernelWidth = 2,
        int kernelHeight = 2)
    {
        var width = matrix.GetLength(0);
        var height = matrix.GetLength(1);

        var result = new int[width / kernelWidth, height / kernelHeight];

        for (int x = 0; x <= width - kernelWidth; x += kernelWidth)
        {
            for (int y = 0; y <= height - kernelHeight; y += kernelHeight)
            {
                var max = 0;
                for (int i = 0; i < kernelWidth; i++)
                {
                    for (int j = 0; j < kernelHeight; j++)
                    {
                        var matrixValue = matrix[x + i, y + j];
                        if (matrixValue > max)
                            max = matrixValue;
                    }
                }

                result[x / kernelWidth, y / kernelHeight] = max;
            }
        }

        return result;
    }

    /// <summary>
    /// Filter Image By Specific Kernel
    /// </summary>
    /// <param name="matrix"></param>
    /// <param name="kernel"></param>
    /// <returns></returns>
    public int[,] Filter(int[,] matrix, float[,] kernel)
    {
        var width = matrix.GetLength(0);
        var height = matrix.GetLength(1);

        var result = new int[width / 3, height / 3];

        for (int x = 0; x <= width - 3; x += 3)
        {
            for (int y = 0; y <= height - 3; y += 3)
            {
                var smallMatrix = new int[3, 3];
                for (int i = 0; i < kernel.GetLength(0); i++)
                {
                    for (int j = 0; j < kernel.GetLength(1); j++)
                    {
                        smallMatrix[i, j] = matrix[x + i, y + j];
                    }
                }

                var output = kernel.Mul(smallMatrix);
                var pixelValue = output.Sum();

                result[x / 3, y / 3] = pixelValue;
            }
        }

        return result;
    }

    /// <summary>
    /// Up Scale Image To Bigger Size
    /// </summary>
    /// <param name="matrix"></param>
    /// <param name="newWidth"></param>
    /// <param name="newHeight"></param>
    /// <returns></returns>
    public int[,] UpscaleImage(int[,] matrix, int newWidth, int newHeight)
    {
        var width = matrix.GetLength(0);
        var height = matrix.GetLength(1);

        var output = new int[newWidth, newHeight];

        var xStep = newWidth / width;
        var yStep = newHeight / height;

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                var pixelValue = matrix[x, y];

                for (int i = 0; i < xStep; i++)
                {
                    for (int ii = 0; ii < yStep; ii++)
                    {
                        output[(x * xStep) + i, (y * yStep) + ii] = pixelValue;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Convert Gray Scale Matrix To Gray Scale Image
    /// </summary>
    /// <param name="matrix"></param>
    /// <returns></returns>
    public Bitmap ConvertToBitmap(int[,] matrix)
    {
        var width = matrix.GetLength(0);
        var height = matrix.GetLength(1);

        Bitmap bitmap = new Bitmap(width, height);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                var color = matrix[x, y];

                if (color < 0)
                    color = -color;

                if (color > 255)
                    color = 255;

                bitmap.SetPixel(x, y, Color.FromArgb(color, color, color));
            }
        }

        return bitmap;
    }

    /// <summary>
    /// Convert RGB Matrix To RGB Image
    /// </summary>
    /// <param name="matrix"></param>
    /// <returns></returns>
    public Bitmap ConvertToBitmap(int[,,] matrix)
    {
        var width = matrix.GetLength(0);
        var height = matrix.GetLength(1);

        Bitmap bitmap = new Bitmap(width, height);

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                var red = matrix[0, x, y];
                var green = matrix[1, x, y];
                var blue = matrix[2, x, y];

                bitmap.SetPixel(x, y, Color.FromArgb(red, green, blue));
            }
        }

        return bitmap;
    }

    /// <summary>
    /// Convert Gray Scale Image To Gray Scale Matrix
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    public int[,] ConvertToGrayScaleMatrix(Bitmap image)
    {
        int[,] matrix = new int[image.Width, image.Height];

        // Loop through each pixel and store the RGB value in the matrix
        for (int x = 0; x < image.Width; x++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                Color pixel = image.GetPixel(x, y);
                int red = pixel.R;
                int green = pixel.G;
                int blue = pixel.B;
                var grayColor = (red + green + blue) / 3;
                matrix[x, y] = grayColor;
            }
        }

        return matrix;
    }

    /// <summary>
    /// Convert RGB Image To RGB Matrix
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    public int[,,] ConvertToRGBMatrix(Bitmap image)
    {
        int[,,] matrix = new int[3, image.Width, image.Height];

        // Loop through each pixel and store the RGB value in the matrix
        for (int x = 0; x < image.Width; x++)
        {
            for (int y = 0; y < image.Height; y++)
            {
                Color pixel = image.GetPixel(x, y);
                int red = pixel.R;
                matrix[0, x, y] = red;

                int green = pixel.G;
                matrix[0, x, y] = green;

                int blue = pixel.B;
                matrix[0, x, y] = blue;
            }
        }

        return matrix;
    }

    /// <summary>
    /// Merge Images And Return One Image As Output
    /// </summary>
    /// <param name="images"></param>
    /// <returns></returns>
    public Image MergeImages(List<Bitmap> images)
    {
        var w = images.First().Width;
        var h = images.First().Height;

        var result = new Bitmap(w, h);

        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                var totalPixelValue = 0;
                foreach (var image in images)
                {
                    var pixelColors = image.GetPixel(i, j);
                    var grayColor = (pixelColors.R + pixelColors.G + pixelColors.B) / 3;
                    totalPixelValue += grayColor;
                }
                if (totalPixelValue > 255)
                    totalPixelValue = 255;

                result.SetPixel(i, j, Color.FromArgb(totalPixelValue, totalPixelValue, totalPixelValue));
            }
        }

        return result;
    }
}
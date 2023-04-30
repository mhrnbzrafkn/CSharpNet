using CSharpNet.Infrastructures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace CSharpNet;

/// <summary>
/// Create Deep Neural Network With Custom Layers And Neurons
/// </summary>
public class DeepNetBuilder
{
    public double LearningRate { get; set; }
    public int[] Layers { get; set; }
    public double[] Biases { get; set; }
    public List<double[]> HiddenValues { get; set; }
    public List<double[,]> Weights { get; set; }

    public DeepNetBuilder(int[] layers, double learningRate)
    {
        LearningRate = learningRate;
        Layers = layers;

        var rnd = new Random();
        Biases = new double[layers.Length];
        for (int i = 0; i < Biases.Length; i++)
        {
            Biases[i] = rnd.NextDouble();
        }

        Weights = new List<double[,]>();
        HiddenValues = new List<double[]>();

        for (int i = 0; i < layers.Length - 1; i++)
        {
            var newWeights = MatrixBuilder.Build(layers[i], layers[i + 1]);
            Weights.Add(newWeights);
        }
    }

    public DeepNetBuilder()
    {
        HiddenValues = new List<double[]>();
        Weights = new List<double[,]>();
    }

    public double[] FeedForward(double[] input)
    {
        HiddenValues = new List<double[]>();

        var hiddenLayerOutput = input.Multiply(Weights[0]).Sumation(Biases[0]);
        HiddenValues.Add(hiddenLayerOutput);

        for (int i = 1; i < Weights.Count; i++)
        {
            hiddenLayerOutput = hiddenLayerOutput.Multiply(Weights[i]).Sumation(Biases[i]);
            if (i < Weights.Count - 1)
                HiddenValues.Add(hiddenLayerOutput);
        }

        return hiddenLayerOutput;
    }

    public double Backpropagate(double[] inputs, double[] expectedOutput)
    {
        var actualOutput = FeedForward(inputs);
        var deltas = expectedOutput.Subtract(actualOutput);
        var TwoDDeltas = deltas.ConvertTo2DMatrix();

        var error = 0.0;
        var deltasSum = deltas.Sum();
        for (int i = 0; i < deltas.Length; i++)
        {
            error += (deltas[i] / deltasSum) * deltas[i];
        }

        // update hidden-output weights and bias
        var biasIndes = Biases.Length - 1;
        for (int i = 0; i < TwoDDeltas.Length; i++)
        {
            Biases[biasIndes] -= LearningRate * TwoDDeltas[0, i];
        }

        for (int i = 0; i < Weights.Last().GetLength(0); i++)
        {
            for (int ii = 0; ii < Weights.Last().GetLength(1); ii++)
            {
                Weights.Last()[i, ii] = CalculateNewWeight(
                    actualOutput[ii],
                    expectedOutput[ii],
                    HiddenValues.Last()[i],
                    Weights.Last()[i, ii]);
            }
        }

        if (Layers.Length <= 3)
        {
            var transposedHiddenWeights = Weights[1].Transpose();
            TwoDDeltas = TwoDDeltas.Multiply(transposedHiddenWeights);

            for (int i = 0; i < TwoDDeltas.Length; i++)
            {
                Biases[1] -= LearningRate * TwoDDeltas[0, i];
            }
        }

        // update hidden weights and bias
        if (Layers.Length > 3)
        {
            var transposedHiddenWeights = Weights[Weights.Count - 1].Transpose();
            TwoDDeltas = TwoDDeltas.Multiply(transposedHiddenWeights);

            var biasIndex = Biases.Length - 2;
            for (int i = 0; i < TwoDDeltas.Length; i++)
            {
                Biases[biasIndex] -= LearningRate * TwoDDeltas[0, i];
            }

            for (int i = Weights.Count - 2; i > 0; i--)
            {
                transposedHiddenWeights = Weights[i].Transpose();
                TwoDDeltas = TwoDDeltas.Multiply(transposedHiddenWeights);

                for (int deltaIndex = 0; deltaIndex < TwoDDeltas.Length; deltaIndex++)
                {
                    Biases[i] -= LearningRate * TwoDDeltas[0, deltaIndex];
                }

                for (int ii = 0; ii < Weights[i].GetLength(0); ii++)
                {
                    for (int iii = 0; iii < Weights[i].GetLength(1); iii++)
                    {
                        Weights[i][ii, iii] = CalculateNewWeight(
                        TwoDDeltas[0, ii],
                        HiddenValues[i - 1][ii],
                        Weights[i][ii, iii]);
                    }
                }
            }
        }

        // update input-hidden weights and bias
        var transposedWeights = Weights.First().Transpose();
        TwoDDeltas = TwoDDeltas.Multiply(transposedWeights);

        for (int i = 0; i < TwoDDeltas.Length; i++)
        {
            Biases[0] -= LearningRate * TwoDDeltas[0, i];
        }

        for (int i = 0; i < Weights.First().GetLength(0); i++)
        {
            for (int ii = 0; ii < Weights.First().GetLength(1); ii++)
            {
                Weights.First()[i, ii] = CalculateNewWeight(
                    TwoDDeltas[0, i],
                    inputs[i],
                    Weights.First()[i, ii]);
            }
        }

        // return error
        return Math.Abs(error) * 100;
    }

    public void Train(List<double[]> inputs, List<double[]> outputs, int iterationNumber, double threshold)
    {
        if (inputs.Count != outputs.Count)
            throw new Exception("Inconsistency Between The Number Of Input And Output Data.");

        if (threshold <= 0 || threshold > 1)
            throw new Exception("Enter Threshold Value Between 0 And 1.");

        var previousError = 0.0;
        for (int i = 1; i <= iterationNumber; i++)
        {
            var error = 0.0;

            for (int ii = 0; ii < inputs.Count; ii++)
            {
                error += Backpropagate(inputs[ii], outputs[ii]);
            }

            var totalError = error / inputs.Count;
            Console.WriteLine($"iteration <{i}> - Error <{totalError}>");

            if (previousError == 0.0)
            {
                previousError = totalError;
            }

            else if (totalError > previousError)
            {
                if (totalError - previousError > threshold * 100)
                {
                    Console.WriteLine("Threshold Crossed.");
                    break;
                }
            }

            previousError = totalError;

            if (totalError < threshold * 10)
            {
                Console.WriteLine("Maximum Optimization For This Initial Reached.");
                break;
            }
        }
    }

    private double CalculateNewWeight(
        double actualOutput,
        double expectedOutput,
        double neuronValue,
        double weightValue)
    {
        var delta = -(expectedOutput - actualOutput) * SigmoidDerivative(neuronValue * weightValue) * neuronValue;
        var newWeight = weightValue - LearningRate * delta;

        return newWeight;
    }

    private double CalculateNewWeight(
        double error,
        double neuronValue,
        double weightValue)
    {
        var delta = -(error) * SigmoidDerivative(neuronValue * weightValue) * neuronValue;
        var newWeight = weightValue - LearningRate * delta;

        return newWeight;
    }

    private double Sigmoid(double value)
    {
        return 1 / (1 + Math.Exp(-value));
    }

    private double SigmoidDerivative(double value)
    {
        double fx = Sigmoid(value);
        return fx * (1 - fx);
    }

    public void Save(string path)
    {
        var model = new JsonDeepNetModel(this);
        var json = JsonSerializer.Serialize(model);
        File.WriteAllText(path, json);
    }

    public static DeepNetBuilder LoadModel(string path)
    {
        var file = File.ReadAllText(path);
        var deepNetModel = JsonSerializer.Deserialize<JsonDeepNetModel>(file);

        if (deepNetModel == null)
            throw new Exception("Couldn't Find Model File..");

        var deepNet = JsonDeepNetModel.Deserialize(deepNetModel);

        return deepNet;
    }
}

public class JsonDeepNetModel
{
    public double LearningRate { get; set; }
    public int[] Layers { get; set; }
    public List<double[]> HiddenValues { get; set; }
    public List<double[][]> Weights { get; set; }

    public JsonDeepNetModel()
    {

    }

    public JsonDeepNetModel(DeepNetBuilder deepNet)
    {
        LearningRate = deepNet.LearningRate;
        Layers = deepNet.Layers; HiddenValues = deepNet.HiddenValues;
        Weights = deepNet.Weights.SerializeToJsonArray();
    }

    public static DeepNetBuilder Deserialize(JsonDeepNetModel model)
    {
        var deepNet = new DeepNetBuilder();

        deepNet.LearningRate = model.LearningRate;
        deepNet.Layers = model.Layers;
        deepNet.HiddenValues = model.HiddenValues;
        deepNet.Weights = model.Weights.DeserializeTo2DArray();

        return deepNet;
    }
}

public class MatrixBuilder
{
    public static double[,] Build(int rows, int cols)
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
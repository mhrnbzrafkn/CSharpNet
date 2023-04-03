using SimpleDeepNet.ExtensionMethods;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace CSharpNet;

public class DeepNetBuilder
{
    public double LearningRate { get; set; }
    public int[] Layers { get; set; }
    public List<double[]> HiddenValues { get; set; }
    public List<double[,]> Weights { get; set; }

    public DeepNetBuilder(int[] layers, double learningRate)
    {
        LearningRate = learningRate;
        Layers = layers;
        Weights = new List<double[,]>();
        HiddenValues = new List<double[]>();

        for (int i = 0; i < layers.Length - 1; i++)
        {
            var newWeights = Matrix.Generate(layers[i], layers[i + 1]);
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

        var hiddenLayerOutput = input.Multiply(Weights[0]);
        HiddenValues.Add(hiddenLayerOutput);

        for (int i = 1; i < Weights.Count; i++)
        {
            hiddenLayerOutput = hiddenLayerOutput.Multiply(Weights[i]);
            if (i < Weights.Count - 1)
                HiddenValues.Add(hiddenLayerOutput);
        }

        return hiddenLayerOutput;
    }

    public void Backpropagate(double[] inputs, double[] expectedOutput)
    {
        var actualOutput = FeedForward(inputs);
        var deltas = expectedOutput.Subtract(actualOutput).ConvertTo2DMatrix();

        // update hidden-output weights
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
            deltas = deltas.Multiply(transposedHiddenWeights);
        }

        // update hidden weights
        if (Layers.Length > 3)
        {
            var transposedHiddenWeights = Weights[Weights.Count - 1].Transpose();
            deltas = deltas.Multiply(transposedHiddenWeights);

            for (int i = Weights.Count - 2; i > 0; i--)
            {
                transposedHiddenWeights = Weights[i].Transpose();
                deltas = deltas.Multiply(transposedHiddenWeights);
                for (int ii = 0; ii < Weights[i].GetLength(0); ii++)
                {
                    for (int iii = 0; iii < Weights[i].GetLength(1); iii++)
                    {
                        Weights[i][ii, iii] = CalculateNewWeight(
                        deltas[0, ii],
                        HiddenValues[i - 1][ii],
                        Weights[i][ii, iii]);
                    }
                }
            }
        }

        // update input-hidden weights
        var transposedWeights = Weights.First().Transpose();
        deltas = deltas.Multiply(transposedWeights);
        for (int i = 0; i < Weights.First().GetLength(0); i++)
        {
            for (int ii = 0; ii < Weights.First().GetLength(1); ii++)
            {
                Weights.First()[i, ii] = CalculateNewWeight(
                    deltas[0, i],
                    inputs[i],
                    Weights.First()[i, ii]);
            }
        }
    }

    private double CalculateNewWeight(
        double actualOutput,
        double expectedOutput,
        double neuronValue,
        double weightValue)
    {
        var delta = -(expectedOutput - actualOutput) * (Sigmoid(neuronValue * weightValue)) * (1 - Sigmoid(neuronValue * weightValue)) * neuronValue;
        var newWeight = weightValue - LearningRate * delta;

        return newWeight;
    }

    private double CalculateNewWeight(
        double error,
        double neuronValue,
        double weightValue)
    {
        var delta = -(error) * (Sigmoid(neuronValue * weightValue)) * (1 - Sigmoid(neuronValue * weightValue)) * neuronValue;
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
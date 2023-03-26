using CSharpNet.FeedForwardNet.ExtensionMethods;
using System;
using System.IO;
using System.Text.Json;

namespace CSharpNet.FeedForwardNet;

public class FeedForwardNet
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }
    public double LearninRate { get; set; }
    public int IterationNumber { get; set; }
    public double[][] WeightsInputHidden { get; set; }
    public double[] BiasesInputHidden { get; set; }
    public double[][] WeightsHiddenOutput { get; set; }
    public double[] BiasesHiddenOutput { get; set; }

    public FeedForwardNet(
        int inputSize,
        int hiddenSize,
        int outputSize,
        double learningRate,
        int iterationNumber)
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        OutputSize = outputSize;
        LearninRate = learningRate;
        IterationNumber = iterationNumber;

        var rnd = new Random();

        WeightsInputHidden = new double[InputSize][];
        for (int i = 0; i < InputSize; i++)
        {
            WeightsInputHidden[i] = new double[HiddenSize];
            for (int ii = 0; ii < HiddenSize; ii++)
            {
                WeightsInputHidden[i][ii] = (rnd.NextDouble() * 2) - 1;
            }
        }

        BiasesInputHidden = new double[hiddenSize];
        for (int biasIndex = 0; biasIndex < hiddenSize; biasIndex++)
        {
            BiasesInputHidden[biasIndex] = (rnd.NextDouble() * 2) - 1;
        }

        WeightsHiddenOutput = new double[HiddenSize][];
        for (int i = 0; i < HiddenSize; i++)
        {
            WeightsHiddenOutput[i] = new double[OutputSize];
            for (int ii = 0; ii < OutputSize; ii++)
            {
                WeightsHiddenOutput[i][ii] = (rnd.NextDouble() * 2) - 1;
            }
        }

        BiasesHiddenOutput = new double[outputSize];
        for (int biasIndex = 0; biasIndex < outputSize; biasIndex++)
        {
            BiasesHiddenOutput[biasIndex] = (rnd.NextDouble() * 2) - 1;
        }
    }

    public FeedForwardNet() { }

    public double Sigmoid(double value)
    {
        return 1 / (1 + Math.Exp(-value));
    }

    public double[] FeedForward(double[] inputs)
    {
        var mulInputMatrixes = inputs.Mul(WeightsInputHidden);
        var hidden = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
        {
            var value = mulInputMatrixes[i] + BiasesInputHidden[i];
            //hidden[i] = Sigmoid(mulInputMatrixes[i]);
            hidden[i] = Sigmoid(value);
        }

        var mulOutputMatrixes = hidden.Mul(WeightsHiddenOutput);
        var output = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            var value = mulOutputMatrixes[i] + BiasesHiddenOutput[i];
            //output[i] = Sigmoid(mulOutputMatrixes[i]);
            output[i] = Sigmoid(value);
        }

        return output;
    }

    public void Backpropagate(double[] inputs, double[] targets)
    {
        //Forward pass
        var mulInputMatrixes = inputs.Mul(WeightsInputHidden);
        var hidden = new double[HiddenSize];
        for (int i = 0; i < HiddenSize; i++)
        {
            var value = mulInputMatrixes[i] + BiasesInputHidden[i];
            //hidden[i] = Sigmoid(mulInputMatrixes[i]);
            hidden[i] = Sigmoid(value);
        }

        var mulOutputMatrixes = hidden.Mul(WeightsHiddenOutput);
        var output = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            var value = mulOutputMatrixes[i] + BiasesHiddenOutput[i];
            //output[i] = Sigmoid(mulOutputMatrixes[i]);
            output[i] = Sigmoid(value);
        }

        //Calculate the error
        var error = targets.Sub(output);

        //Backward pass
        var outputDelta = error.Mul(output).Mul(1.Sub(output));
        var hiddenError = outputDelta.Mul(WeightsHiddenOutput.T());
        var hiddenDelta = hiddenError.Mul(hidden).Mul(1.Sub(hidden));

        //Update the weights and biases
        var outer1 = hidden.outer(outputDelta);
        var newOuter1 = LearninRate.Dot(outer1);
        WeightsHiddenOutput = WeightsHiddenOutput.Sum(newOuter1);

        var BiasesHiddenOutputIncrease = LearninRate.Dot(outputDelta);
        BiasesHiddenOutput = BiasesHiddenOutput.Sub(BiasesHiddenOutputIncrease);

        var outer2 = inputs.outer(hiddenDelta);
        var newOuter2 = LearninRate.Dot(outer2);
        WeightsInputHidden = WeightsInputHidden.Sum(newOuter2);

        var BiasesInputHiddenIncrease = LearninRate.Dot(hiddenDelta);
        BiasesInputHidden = BiasesInputHidden.Sub(BiasesInputHiddenIncrease);
    }

    public void Save(string path)
    {
        string json = JsonSerializer.Serialize(this);
        File.WriteAllText(path, json);
    }

    public static FeedForwardNet LoadModel(string path)
    {
        var file = File.ReadAllText(path);
        var nn = JsonSerializer.Deserialize<FeedForwardNet>(file);

        if (nn == null)
            throw new Exception("Couldn't Find Model File..");

        return nn;
    }
}
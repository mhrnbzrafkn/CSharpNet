using CSharpNet.FeedForwardNet.ExtensionMethods;
using System.Linq;
using System.Text.Json;
using System;
using System.IO;

namespace CSharpNet.FeedForwardNet
{
    public class DeepNet
    {
        public int InputSize { get; set; }
        public int[] HiddenSizes { get; set; }
        public int OutputSize { get; set; }

        public double LearningRate { get; set; }
        public int IterationNumber { get; set; }

        public double[][] WeightsInputHidden { get; set; }
        public double[] BiasesInputHidden { get; set; }

        public double[][][] WeightsHidden { get; set; }
        public double[][] BiasesHidden { get; set; }

        public double[][] WeightsHiddenOutput { get; set; }
        public double[] BiasesHiddenOutput { get; set; }

        public DeepNet(
            int inputSize,
            int[] hiddenSizes,
            int outputSize,
            double learningRate,
            int iterationNumber)
        {
            InputSize = inputSize;
            HiddenSizes = new int[hiddenSizes.Length];
            for (int i = 0; i < hiddenSizes.Length; i++)
            {
                HiddenSizes[i] = hiddenSizes[i];
            }
            OutputSize = outputSize;
            LearningRate = learningRate;
            IterationNumber = iterationNumber;
            // ----- \\
            var rnd = new Random();
            WeightsInputHidden = new double[InputSize][];
            for (int i = 0; i < InputSize; i++)
            {
                WeightsInputHidden[i] = new double[HiddenSizes.First()];
                for (int ii = 0; ii < HiddenSizes.First(); ii++)
                {
                    WeightsInputHidden[i][ii] = (rnd.NextDouble() * 2) - 1;
                }
            }

            BiasesInputHidden = new double[HiddenSizes.First()];
            for (int i = 0; i < HiddenSizes.First(); i++)
            {
                BiasesInputHidden[i] = (rnd.NextDouble() * 2) - 1;
            }
            // ----- \\
            WeightsHidden = new double[HiddenSizes.Length - 1][][];
            for (int i = 0; i < HiddenSizes.Length - 1; i++)
            {
                WeightsHidden[i] = new double[HiddenSizes[i]][];
                for (int ii = 0; ii < HiddenSizes[i]; ii++)
                {
                    WeightsHidden[i][ii] = new double[HiddenSizes[i + 1]];
                    for (int iii = 0; iii < HiddenSizes[i + 1]; iii++)
                    {
                        WeightsHidden[i][ii][iii] = (rnd.NextDouble() * 2) - 1;
                    }
                }
            }

            BiasesHidden = new double[HiddenSizes.Length - 1][];
            for (int i = 0; i < HiddenSizes.Length - 1; i++)
            {
                BiasesHidden[i] = new double[HiddenSizes[i + 1]];
                for (int ii = 0; ii < HiddenSizes[i + 1]; ii++)
                {
                    BiasesHidden[i][ii] = (rnd.NextDouble() * 2) - 1;
                }
            }
            // ----- \\
            WeightsHiddenOutput = new double[HiddenSizes.Last()][];
            for (int i = 0; i < HiddenSizes.Last(); i++)
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
            // ----- \\
        }

        public double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        public double[] FeedForward(double[] inputs)
        {
            var mulMatrixes = inputs.Mul(WeightsInputHidden);
            var firstHidden = new double[HiddenSizes.First()];
            for (int i = 0; i < HiddenSizes.First(); i++)
            {
                var value = mulMatrixes[i] + BiasesInputHidden[i];
                firstHidden[i] = Sigmoid(value);
            }

            // calculate hiddens
            var hiddens = new double[HiddenSizes.Length][];
            hiddens[0] = firstHidden;
            for (int i = 0; i < HiddenSizes.Length - 1; i++)
            {
                var hidden = new double[HiddenSizes[i + 1]];
                for (int ii = 0; ii < HiddenSizes[i + 1]; ii++)
                {
                    var mul = hiddens[i].Mul(WeightsHidden[i]);
                    var value = mul[i] + BiasesHidden[i][ii];
                    hidden[ii] = Sigmoid(value);
                }
                hiddens[i + 1] = hidden;
            }

            // calculate outputs
            var mulOutputMatrixes = hiddens.Last().Mul(WeightsHiddenOutput);
            var output = new double[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                var value = mulOutputMatrixes[i] + BiasesHiddenOutput[i];
                output[i] = Sigmoid(value);
            }

            return output;
        }

        public void Backpropagate(double[] inputs, double[] targets)
        {
            // feedforward
            // calculate inputs
            var mulMatrixes = inputs.Mul(WeightsInputHidden);
            var firstHidden = new double[HiddenSizes.First()];
            for (int i = 0; i < HiddenSizes.First(); i++)
            {
                var value = mulMatrixes[i] + BiasesInputHidden[i];
                firstHidden[i] = Sigmoid(value);
            }

            // calculate hiddens
            var hiddens = new double[HiddenSizes.Length][];
            hiddens[0] = firstHidden;
            for (int i = 0; i < HiddenSizes.Length - 1; i++)
            {
                var hidden = new double[HiddenSizes[i + 1]];
                for (int ii = 0; ii < HiddenSizes[i + 1]; ii++)
                {
                    var mul = hiddens[i].Mul(WeightsHidden[i]);
                    var value = mul[i] + BiasesHidden[i][ii];
                    hidden[ii] = Sigmoid(value);
                }
                hiddens[i + 1] = hidden;
            }

            // calculate outputs
            var mulOutputMatrixes = hiddens.Last().Mul(WeightsHiddenOutput);
            var output = new double[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                var value = mulOutputMatrixes[i] + BiasesHiddenOutput[i];
                output[i] = Sigmoid(value);
            }

            // Calculate the error
            var outputError = targets.Sub(output);

            // Backward pass
            var outputDelta = outputError.Mul(output).Mul(1.Sub(output));

            var hiddenErrors = new double[HiddenSizes.Length][];
            hiddenErrors[HiddenSizes.Length - 1] = outputDelta.Mul(WeightsHiddenOutput.T());
            var hiddenDeltas = new double[HiddenSizes.Length][];
            hiddenDeltas[HiddenSizes.Length - 1] = hiddenErrors[HiddenSizes.Length - 1].Mul(hiddens.Last()).Mul(1.Sub(hiddens.Last()));

            for (int i = HiddenSizes.Length - 2; i >= 0; i--)
            {
                hiddenErrors[i] = hiddenDeltas[i + 1].Mul(WeightsHidden[i].T());
                hiddenDeltas[i] = hiddenErrors[i].Mul(hiddens[i]).Mul(1.Sub(hiddens[i]));
            }

            // Update the weights and biases
            // update output weights
            var outputOuter = hiddens.Last().outer(outputDelta);
            var newOutputOuter = LearningRate.Dot(outputOuter);
            WeightsHiddenOutput = WeightsHiddenOutput.Sum(newOutputOuter);

            var biasesHiddenOutputIncrease = LearningRate.Dot(outputDelta);
            BiasesHiddenOutput = BiasesHiddenOutput.Sub(biasesHiddenOutputIncrease);

            //  hidden weights
            for (int i = HiddenSizes.Length - 2; i >= 0; i--)
            {
                var hiddenOuter = hiddens[i].outer(hiddenDeltas[i + 1]);
                var newHiddenOuter = LearningRate.Dot(hiddenOuter);
                WeightsHidden[i] = WeightsHidden[i].Sum(newHiddenOuter);
            }

            // update input weights
            var inputOuter = inputs.outer(hiddenDeltas.First());
            var newInputOuter = LearningRate.Dot(inputOuter);
            WeightsInputHidden = WeightsInputHidden.Sum(newInputOuter);

            var biasesInputHiddenIncrease = LearningRate.Dot(hiddenDeltas.First());
            BiasesInputHidden = BiasesInputHidden.Sub(biasesInputHiddenIncrease);
        }

        public void Save(string path)
        {
            string json = JsonSerializer.Serialize(this);
            File.WriteAllText(path, json);
        }

        public static DeepNet LoadModel(string path)
        {
            var file = File.ReadAllText(path);
            var nn = JsonSerializer.Deserialize<DeepNet>(file);

            if (nn == null)
                throw new Exception("Couldn't Find Model File..");

            return nn;
        }
    }
}

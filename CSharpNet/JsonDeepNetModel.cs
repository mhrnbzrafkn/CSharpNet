using SimpleDeepNet.ExtensionMethods;
using System.Collections.Generic;

namespace CSharpNet;

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
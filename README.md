
# CSharpNet
simple implementation of [Neural Network](https://medium.com/analytics-vidhya/what-is-a-neural-network-and-how-do-they-work-61b38d2720b8#:~:text=Neural%20networks%20are%20computational%20systems,of%20what%20the%20data%20is.) with C#.

## Descriptions
With this library you can easily make a neural network and train it with data and predict output values.
- Note that this is very simple library and does not offer all the features that other libraries have.
- Obviously, the accuracy and efficiency of the neural network you build is completely dependent on the quality of the data you use.
 
## Usage
### Installation
You can search the package name in `NuGet` Manager  ([CSharpNet](https://www.nuget.org/packages/CSharpNet)), or use the commands below in package manager console.
The installation contains two parts, the first is the main body:

```sh
### Install CSharpNet
PM> Install-Package CSharpNet
```
Two simple examples are given here to introduce the basic usage of `CSharpNet`. As you can see, it's easy to write C# code.
#
### Road map to work with library:
- Create a neural network with 3 layers.
- - First layer has 2 neurons
- - Second layer has 3 neurons
- - Third layer has 1 neuron (which is final output of neural network)
- `FeedForward` operation
- - this method can use for prediction.
- `Backpropagate` operation
- `Train` operation
- `Save` model
- `load` model

#
### Arrange variables
- Create an array of type `int` to define layers and their neurons:
```csharp
var layers = new int[3] { 2, 3, 1 };
```
- Create a variable to define learning rate of neural network: 
```csharp
var learningRate = 0.1;
```
- Create a variable to define activation function of neural network layers:
```csharp
var activationFunction = ActivationFunction.Sigmoid;
```
- Create an instance of class `DeepNetBuilder`
```csharp
var nn = new DeepNetBuilder(layers, learningRate, activationFunction);
```
Now you have a neural network, ready to learn everything you want.

#
### Feed data to neural network
Now you can use `FeedForward` method and feed your input(s) data to neural network for calculate output(s).
- Predict output(s) with specific input(s)
```csharp
var inputs = new double[2] { 1, 0 };
var output = nn.FeedForward(inputs);
```

#
### Backpropagate operation
This operation is the main part of train operation.
In this example:
- the number of inputs should be 2 because the number of neural network inputs is 2.
- And the number of expected outputs should be 1 because the number of neural network outputs is 1.
```csharp
var inputs = new double[2] { 1, 0 };
var expectedOutput = new double[1] { 1 };
nn.Backpropagate(inputs, expectedOutput);
```

#
### Train operation
Unlike the previous method, this method takes 2 list of data as input, each record in the first list is an array of type double with length equal to the number of the neurons in the input layer of the neural network, The second list length should be equal to the first list and each record of the list is an array of type double with length equal to the number of the neurons in the output layer of the neural network.
```csharp
var inputs = new List<double[]>()
	{
		new double[2] { 1, 1 },
        	new double[2] { 2, 2 },
        	new double[2] { 3, 3 },
	};
var targets = new List<double[]>()
	{
		new double[1] { 2 },
        	new double[1] { 4 },
        	new double[1] { 6 },
	};
	
nn.Train(inputs, targets, iterationNumber: 1000, threshold: 0.1);
```
- This method run train on the data for 1000 iterations and threshold shows acceptable error value.

#
### Save model
To save model and use it again after a while, you can use `Save` method.
It works very easy and you should give it a path to save model as (`json`) file.
- Note that the path should contain file name and (`json`) extension.
```csharp
var filePath = ".\\Models\\carTest.json";
nn.Save(filePath);
```
#
### Load model
To load load model for `use` or `train` it again, you can use `LoadModel` method.
It loads and cast (`json`) file to an object of type `JsonDeepNetModel` as you can see in the repository.
- Note that the path should contain file name and (`json`) extension.
```csharp
var filePath = ".\\Models\\carTest.json";
var nn = DeepNetBuilder.LoadModel(filePath );
```
#
### Contact
Follow me on [LinkedIn](https://www.linkedin.com/in/mehran-bazrafkan/).

# geneticNN - Genetic algorithm for Neural Network optimization

geneticNN utilizes genetic programming to automatically architect a deep neural network with optimal hyperparameters for a given dataset using the TensorFlow library. This approach should design an equal or superior model to what a human could design when working under the same constraints as are imposed upon the genetic program (e.g., maximum number of layers, maximum number of neurons in each layer, etc.). The current setup is designed for prediction problems, though this could be extended to include any other output type as well.

See `CalcFold/demo.py` for a simple example.

## Evolution

Each model is represented as fixed-width genome encoding information about the network's structure. In the current setup, a model contains a number of LSTM layers and a value describing the amount of epochs to use for training. The complexity of these models could easily be extended beyond these capabilities to include any parameters included in TensorFlow, allowing the creation of more complex architectures.

Below is a highly simplified visualization of how genetic crossover might take place between two models.

<img width="75%" src="https://preview.ibb.co/gdMDak/crossover.png">
<i>Genetic crossover and mutation of neural networks</i>

## Application

The most significant barrier in using geneticNN on a real problem is the complexity of the algorithm. Because training neural networks is often such a computationally expensive process, training hundreds or thousands of different models to evaluate the fitness of each is not always feasible. Below are some approaches to combat this issue:

- **Parallel Training** - The nature of evaluating the fitness of multiple members of a population simultaneously is *embarassingly parallel*. A task like this would be trivial to distribute among many GPUs and even machines.
- **Early Stopping** - There's no need to train a model for 10 epochs if it stops improving after 3; cut it off early.
- **Train on Fewer Epochs** - Training in a genetic program serves one purpose: to evaluate a model's fitness in relation to other models. It may not be necessary to train to convergence to make this comparison; you may only need 2 or 3 epochs. However, it is important you exercise caution in decreasing training time because doing so could create evolutionary pressure toward simpler models that converge quickly. This creates a trade-off between training time and accuracy which, depending on the application, may or may not be desirable. 
- **Parameter Selection** - The more robust you allow your models to be, the longer it will take to converge; i.e., don't allow horizontal flipping on a character recognition problem even though the genetic program will eventually learn not to include it. The less space the program has to explore, the faster it will arrive at an optimal solution. 

## Wanna Try It?

To setup, just clone the repo and run `pip install -e path/to/repo`. You should then be able to access geneticNN globally.

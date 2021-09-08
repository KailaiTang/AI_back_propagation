import java.util.*;

/**
 * The main class that handles the entire network Has multiple attributes each with its own use
 */

public class NNImpl {
  private ArrayList<Node> inputNodes; // list of the output layer nodes.
  private ArrayList<Node> hiddenNodes; // list of the hidden layer nodes
  private ArrayList<Node> outputNodes; // list of the output layer nodes

  private ArrayList<Instance> trainingSet; // the training set

  private double learningRate; // variable to store the learning rate
  private int maxEpoch; // variable to store the maximum number of epochs
  private Random random; // random number generator to shuffle the training set

  /**
   * This constructor creates the nodes necessary for the neural network Also connects the nodes of
   * different layers After calling the constructor the last node of both inputNodes and hiddenNodes
   * will be bias nodes.
   */

  NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount,
      Double learningRate, int maxEpoch, Random random,
      Double[][] hiddenWeights, Double[][] outputWeights) {
    this.trainingSet = trainingSet;
    this.learningRate = learningRate;
    this.maxEpoch = maxEpoch;
    this.random = random;

    // input layer nodes
    inputNodes = new ArrayList<>();
    int inputNodeCount = trainingSet.get(0).attributes.size();
    int outputNodeCount = trainingSet.get(0).classValues.size();
    for (int i = 0; i < inputNodeCount; i++) {
      Node node = new Node(0);
      inputNodes.add(node);
    }

    // bias node from input layer to hidden
    Node biasToHidden = new Node(1);
    inputNodes.add(biasToHidden);

    // hidden layer nodes
    hiddenNodes = new ArrayList<>();
    for (int i = 0; i < hiddenNodeCount; i++) {
      Node node = new Node(2);
      // Connecting hidden layer nodes with input layer nodes
      for (int j = 0; j < inputNodes.size(); j++) {
        NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j),
            hiddenWeights[i][j]);
        node.parents.add(nwp);
      }
      hiddenNodes.add(node);
    }

    // bias node from hidden layer to output
    Node biasToOutput = new Node(3);
    hiddenNodes.add(biasToOutput);

    // Output node layer
    outputNodes = new ArrayList<>();
    for (int i = 0; i < outputNodeCount; i++) {
      Node node = new Node(4);
      // Connecting output layer nodes with hidden layer nodes
      for (int j = 0; j < hiddenNodes.size(); j++) {
        NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j),
            outputWeights[i][j]);
        node.parents.add(nwp);
      }
      outputNodes.add(node);
    }
  }

  /**
   * Get the prediction from the neural network for a single instance Return the idx with highest
   * output values. For example if the outputs of the outputNodes are [0.1, 0.5, 0.2], it should
   * return 1. The parameter is a single instance
   */
  public int predict(Instance instance) {
    forward(instance);
    // find out the maximum value among three output nodes, select the largest one as prediction result
    int maxIndex = 0;
    double max = outputNodes.get(0).getOutput();;
    for (int i = 0; i < outputNodes.size(); i++) {
      if (outputNodes.get(i).getOutput() > max) {
        maxIndex = i;
        max = outputNodes.get(i).getOutput();
      }
    }
    return maxIndex;
  }

  /**
   * private helper method to set the network to do the forward propogation
   * 
   * @param instance
   */
  private void forward(Instance instance) {
    // put the instance's attributes to nodes in input layers
    for (int i = 0; i < inputNodes.size() - 1; i++) { // -1 because the bias node is excluded
      Node input = inputNodes.get(i);
      input.setInput(instance.attributes.get(i));
    }

    // calculate the output generated for hidden layer
    for (int i = 0; i < hiddenNodes.size() - 1; i++) {
      Node hidden = hiddenNodes.get(i);
      hidden.calculateOutput(0);
      // hidden.calculateDelta();
    }

    // calculate the softMaxSum
    double softMaxSum = 0.0;
    for (Node e : outputNodes) {
      double cur = 0;
      for (NodeWeightPair i : e.parents) {
        cur += i.node.getOutput() * i.weight;
      }
      softMaxSum += Math.pow(Math.E, cur);
    }

    // calculate the output generated for output layer
    for (Node e : outputNodes) {
      e.calculateOutput(softMaxSum);
    }
  }



  /**
   * Train the neural networks with the given parameters
   * <p>
   * The parameters are stored as attributes of this class
   */
  public void train() {



    for (int epoch = 0; epoch < maxEpoch; epoch++) {
      // update the totalLoss to 0
      double totalLoss = 0;

      // shuffle
      Collections.shuffle(trainingSet, random);

      // update the total loss to 0
      totalLoss = 0;

      // repreat until the epoch reaches the maxEpoch
      for (Instance e : trainingSet) {
        forward(e);
        back(e);
      }

      // update the totalLoss when all instance has been forward and back once
      for (Instance e : trainingSet) {
        totalLoss += loss(e);
      }

      double averageLoss = totalLoss / trainingSet.size();

      System.out.print("Epoch: " + epoch + ", Loss: ");
      System.out.printf("%.3e", averageLoss);
      System.out.println();

    }


  }

  /**
   * helper method to to the back propogation to update the weight
   * 
   * @param instance
   */
  private void back(Instance instance) {
    updateDelta(instance);
    for (Node n : outputNodes) {
      n.updateWeight(learningRate);
    }
    for (Node n : hiddenNodes) {
      n.updateWeight(learningRate);
    }
  }

  private void updateDelta(Instance instance) {
    // 1. if it is an output node
    for (int i = 0; i < outputNodes.size(); i++) {
      double delta = instance.classValues.get(i)
          - outputNodes.get(i).getOutput();
      outputNodes.get(i).calculateDelta(delta);
    }
    // 2. if it is a hidden node
    for (int i = 0; i < hiddenNodes.size() - 1; i++) { // exclude the bias node
      Node hidden = hiddenNodes.get(i);
      double sigma = 0;
      for (Node e : outputNodes) {
        sigma += e.parents.get(i).weight * e.getDelta();
      }
      if (hidden.getOutput() <= 0) {
        hidden.calculateDelta(0);
      } else {
        hidden.calculateDelta(sigma);
      }
    }
  }

  /**
   * Calculate the cross entropy loss from the neural network for a single instance. The parameter is
   * a single instance
   */
  private double loss(Instance instance) {
    // the cross-entropy loss function for a single example xi is defined as
    // L = the sum of (yk * ln g(zk)) for all K
    // where yi is the target class value
    forward(instance);
    double sumOfTargetAndLnG = 0;
    for (int i = 0; i < outputNodes.size(); i++) {
      // System.out.println("Test: "+n.getOutput());
      sumOfTargetAndLnG -= instance.classValues.get(i)
          * Math.log(outputNodes.get(i).getOutput());
    }
    return sumOfTargetAndLnG;
  }
}

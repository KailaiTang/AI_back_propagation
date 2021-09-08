import java.util.*;

/**
 * Class for internal organization of a Neural Network. There are 5 types of nodes. Check the type
 * attribute of the node for details. Feel free to modify the provided function signatures to fit
 * your own implementation
 */

public class Node {
  private int type = 0; // 0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
  public ArrayList<NodeWeightPair> parents = null; // Array List that will contain the parents (including the bias node) with weights if applicable

  public double inputValue = 0.0;
  private double outputValue = 0.0;
  private double delta = 0.0; // input gradient
  private double outputGradient = 0.0;

  private int target = -1; // the target output (only for output node and has value 0 or 1)
  private ArrayList<Node> outputNodes; // all nodes in the output layers

  // Create a node with a specific type
  Node(int type) {
    if (type > 4 || type < 0) {
      System.out.println("Incorrect value for node type");
      System.exit(1);

    } else {
      this.type = type;
    }
    if (type == 2 || type == 4) {
      parents = new ArrayList<>();
    }
  }

  /**
   * Setters for input value
   * 
   * @param inputValue
   */
  public void setInput(double inputValue) {
    if (type == 0) { // If input node
      this.inputValue = inputValue;
    }
  }

  /**
   * Calculate the output of a node. You can get this value by using getOutput()
   */
  public void calculateOutput(double softMaxSum) {
    if (type == 2 || type == 4) {
      double weightedSum = 0.0;
      // 3. hidden node
      if (type == 2) {
        // the activation function for the hidden layer is ReLU function, which g(z) = max(0,z)
        // calculate the weighted sum of all inputs in the current node
        for (NodeWeightPair e : parents) {
          weightedSum += e.node.getOutput() * e.weight;
        }
        outputValue = Math.max(0.0, weightedSum);
      }
      // 4. output node
      if (type == 4) {
        // the activation function for the output layer is softmax function,
        // which g(z) = g(zj) = (e^zj) / (sum of e^zk for all k)
        for (NodeWeightPair e : parents) {
          weightedSum += e.node.getOutput() * e.weight;
        }
        outputValue = Math.pow(Math.E, weightedSum) / softMaxSum;
      }

    }

  }

  /**
   * Methods to get the output value
   * 
   * @return
   */
  public double getOutput() {
    // 1. Input node
    if (type == 0) {
      return inputValue;
    }
    // 2. Bias node
    else if (type == 1 || type == 3) {
      return 1.00;
    }
    // 3.4. hidden node and output node
    return outputValue;
  }


  public double getDelta() {
    return this.delta;
  }

  /**
   * Calculate the delta value of a node.
   * 
   */
  public void calculateDelta(double delta) {
    if (type == 2 || type == 4) {
      this.delta = delta;
    }
  }

  public double setDelta(double delta) {
    return delta;
  }

  /**
   * Update the weights between parents node and current node
   * 
   * @param learningRate
   */
  public void updateWeight(double learningRate) {
    if (type == 2 || type == 4) {
      for (NodeWeightPair e : parents) {
        e.weight += learningRate * e.node.getOutput() * getDelta();
      }
    }
  }

}



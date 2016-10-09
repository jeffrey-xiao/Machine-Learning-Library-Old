using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public class Network {
        public List<Layer> layers;
        public double correct, total, totalCost, iterations;
        public Queue<bool> correctList;
        public Queue<double> costList;

        public Network () {
            layers = new List<Layer>();
            correctList = new Queue<bool>();
            costList = new Queue<double>();
        }

        public void addLayer (Layer layer) {
            if (layers.Count > 0) {
                Layer prevLayer = layers.Last();
                layer.BindTo(ref prevLayer);
            }
            layers.Add(layer);
        }

        // 1D input
        public void train (double[,] input, int answer, double learningRate) {
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,,] ret = new double[1, width, height];

            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    ret[0, i, j] = input[i, j];

            train(ret, answer, learningRate);
        }

        // 2D input
        public void train (double[] input, int answer, double learningRate) {
            int size = input.GetLength(0);
            double[,,] ret = new double[1, 1, size];

            for (int i = 0; i < size; i++)
                ret[0, 0, i] = input[i];

            train(ret, answer, learningRate);
        }

        // 1D input
        public int predict (double[] input) {
            int size = input.GetLength(0);
            double[,,] ret = new double[1, 1, size];
            for (int i = 0; i < size; i++)
                ret[0, 0, i] = input[i];

            return predict(ret);
        }

        // 2D input
        public int predict (double[,] input) {
            int width = input.GetLength(0);
            int height = input.GetLength(1);

            double[,,] ret = new double[1, width, height];

            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    ret[0, i, j] = input[i, j];

            return predict(ret);
        }

        public int predict (double[,,] input) {
            List<double[,,]> activatedOutputs = new List<double[,,]>();
            int outputSize = ((Layer1D)layers.Last()).size;;

            for (int i = 0; i < layers.Count; i++) {
                if (i == 0)
                    activatedOutputs.Add(layers[i].forwardPropagate(input));
                else
                    activatedOutputs.Add(layers[i].forwardPropagate(activatedOutputs[i - 1]));
            }

            return maxIndex(activatedOutputs.Last());
        }

        public void train (double[,,] input, int answer, double learningRate) {
            total++;
            iterations++;

            List<double[,,]> activatedOutputs = new List<double[,,]>();
            int outputSize = ((Layer1D)layers.Last()).size;
            double[,,] error = new double[1, 1, outputSize];

            for (int i = 0; i < outputSize; i++)
                error[0, 0, i] = (answer == i ? 1 : -1);

            for (int i = 0; i < layers.Count; i++) {
                if (i == 0)
                    activatedOutputs.Add(layers[i].forwardPropagate(input));
                else
                    activatedOutputs.Add(layers[i].forwardPropagate(activatedOutputs[i - 1]));
            }

            for (int i = layers.Count - 1; i >= 0; i--) {
                error = layers[i].backPropagate(activatedOutputs[i], error, learningRate);
                if (i == layers.Count - 1) {
                    if (maxIndex(activatedOutputs[i]) == answer) {
                        correct++;
                        correctList.Enqueue(true);
                    } else {
                        correctList.Enqueue(false);
                    }

                    double cost = 0;
                    for (int j = 0; j < outputSize; j++) {
                        double currCost = error[0, 0, j] / (1 - activatedOutputs[i][0, 0, j] * activatedOutputs[i][0, 0, j] + 1e-8);
                        cost += currCost * currCost;
                    }
                    totalCost += cost;
                    costList.Enqueue(cost);
                }
            }
        }

        public int maxIndex (double[,,] output) {
            int max = 0;
            for (int i = 1; i < output.GetLength(2); i++)
                if (output[0, 0, i] > output[0, 0, max])
                    max = i;
            return max;
        }
    }
}

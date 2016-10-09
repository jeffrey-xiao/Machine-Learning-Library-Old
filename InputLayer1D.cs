using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {

    public class InputLayer1D : Layer1D {
        public InputLayer1D (int size) {
            this.size = size;
        }

        // input layers will never have a previous layer
        public override void BindTo (ref Layer layer) {
            prevLayer = null;
        }

        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[1, 1, size];

            Debug.Assert(ret.GetLength(0) == prevActivated.GetLength(0));
            Debug.Assert(ret.GetLength(1) == prevActivated.GetLength(1));
            Debug.Assert(ret.GetLength(2) == prevActivated.GetLength(2));

            for (int i = 0; i < size; i++)
                ret[0, 0, i] = prevActivated[0, 0, i];

            return ret;
        }

        // back propagation ends at the input layer
        public override double[,,] backPropagate (double[,,] currActivated, double[,,] nextError, double learningRate) {

            if (nextLayer is Layer1D) {
                Layer1D next = (Layer1D)nextLayer;

                for (int i = 0; i < size; i++)
                    for (int j = 0; j < next.size; j++)
                        next.matrix.weight[0, 0, i, j] -= currActivated[0, 0, i] * nextError[0, 0, j] * learningRate;

                for (int i = 0; i < next.size; i++)
                    next.matrix.bias[i] -= nextError[0, 0, i] * learningRate;
            } else if (nextLayer is Layer2D) {
                throw new Exception("Invalid Layer");
            }

            return null;
        }
    }

    public class FullyConnectedLayer : Layer1D {

        public FullyConnectedLayer (int size) {
            this.size = size;
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;
            layer.nextLayer = this;

            matrix = new NeuronMatrix(layer, this);
        }

        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[1, 1, size];

            for (int i = 0; i < size; i++)
                ret[0, 0, i] = matrix.bias[i];

            if (prevLayer is Layer1D) {
                Layer1D prev = (Layer1D)prevLayer;

                for (int i = 0; i < prev.size; i++)
                    for (int j = 0; j < size; j++)
                        ret[0, 0, j] += matrix.weight[0, 0, i, j] * prevActivated[0, 0, i];

                for (int i = 0; i < size; i++)
                    ret[0, 0, i] = Math.Tanh(ret[0, 0, i]);
            } else if (prevLayer is Layer2D) {
                Layer2D prev = (Layer2D)prevLayer;

                for (int i = 0; i < prev.depth; i++)
                    for (int j = 0; j < prev.width; j++)
                        for (int k = 0; k < prev.height; k++)
                            for (int l = 0; l < size; l++)
                                ret[0, 0, l] += matrix.weight[i, j, k, l] * prevActivated[i, j, k];

                for (int i = 0; i < size; i++)
                    ret[0, 0, i] = Math.Tanh(ret[0, 0, i]);
            }

            return ret;
        }

        // Previous layer is 1D
        public override double[,,] backPropagate (double[,,] currActivated, double[,,] nextError, double learningRate) {
            double[,,] ret = new double[1, 1, size];

            if (nextLayer is Layer1D) {
                Layer1D next = (Layer1D)nextLayer;

                for (int i = 0; i < size; i++)
                    for (int j = 0; j < next.size; j++)
                        ret[0, 0, i] += nextError[0, 0, j] * next.matrix.weight[0, 0, i, j] * (1 - currActivated[0, 0, i] * currActivated[0, 0, i]);

                for (int i = 0; i < size; i++)
                    for (int j = 0; j < next.size; j++)
                        next.matrix.weight[0, 0, i, j] -= currActivated[0, 0, i] * nextError[0, 0, j] * learningRate;

                for (int i = 0; i < next.size; i++)
                    next.matrix.bias[i] -= nextError[0, 0, i] * learningRate;
            } else if (nextLayer is Layer2D) {
                throw new Exception("Invalid Layer");
            }

            return ret;
        }
    }

    public class OutputLayer : Layer1D {

        public OutputLayer (int size) {
            this.size = size;
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;
            layer.nextLayer = this;

            matrix = new NeuronMatrix(layer, this);
        }
        
        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[1, 1, size];

            for (int i = 0; i < size; i++)
                ret[0, 0, i] = matrix.bias[i];

            if (prevLayer is Layer1D) {
                Layer1D prev = (Layer1D)prevLayer;

                for (int i = 0; i < prev.size; i++)
                    for (int j = 0; j < size; j++)
                        ret[0, 0, j] += matrix.weight[0, 0, i, j] * prevActivated[0, 0, i];

                for (int i = 0; i < size; i++)
                    ret[0, 0, i] = Math.Tanh(ret[0, 0, i]);
            }
            if (prevLayer is Layer2D) {
                Layer2D prev = (Layer2D)prevLayer;

                for (int i = 0; i < prev.depth; i++)
                    for (int j = 0; j < prev.width; j++)
                        for (int k = 0; k < prev.height; k++)
                            for (int l = 0; l < size; l++)
                                ret[0, 0, l] += matrix.weight[i, j, k, l] * prevActivated[i, j, k];

                for (int i = 0; i < size; i++)
                    ret[0, 0, i] = Math.Tanh(ret[0, 0, i]);
            }

            return ret;
        }
        
        public override double[,,] backPropagate (double[,,] currActivated, double[,,] nextError, double learningRate) {
            double[,,] ret = new double[1, 1, size];

            for (int i = 0; i < size; i++)
                ret[0, 0, i] = (currActivated[0, 0, i] - nextError[0, 0, i]) * (1 - currActivated[0, 0, i] * currActivated[0, 0, i] + 1e-8);

            return ret;
        }
    }
}

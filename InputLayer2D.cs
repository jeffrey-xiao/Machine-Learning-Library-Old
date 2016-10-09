using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public class InputLayer2D : Layer2D {
        public InputLayer2D (int depth, int width, int height) {
            this.depth = depth;
            this.width = width;
            this.height = height;
            this.type = Layer.LAYER_2D;
        }

        // input layers will never have a previous layer
        public override void BindTo (ref Layer layer) {
            prevLayer = null;
        }

        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[depth, width, height];

            Debug.Assert(ret.GetLength(0) == prevActivated.GetLength(0));
            Debug.Assert(ret.GetLength(1) == prevActivated.GetLength(1));
            Debug.Assert(ret.GetLength(2) == prevActivated.GetLength(2));

            for (int i = 0; i < depth; i++)
                for (int j = 0; j < width; j++)
                    for (int k = 0; k < height; k++)
                        ret[i, j, k] = prevActivated[i, j, k];

            return ret;
        }

        public override double[,,] backPropagate (double[,,] currActivated, double[,,] nextError, double learningRate) {

            if (nextLayer is Layer1D) {
                Layer1D next = (Layer1D)nextLayer;

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < width; i++)
                        for (int j = 0; j < height; j++)
                            for (int k = 0; k < next.size; k++)
                                next.matrix.weight[d, i, j, k] -= currActivated[d, i, j] * nextError[0, 0, k] * learningRate;

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < next.size; i++)
                        next.matrix.bias[d] -= nextError[0, 0, i] * learningRate;
            } else if (nextLayer is ConvolutionalLayer) {
                ConvolutionalLayer next = (ConvolutionalLayer)nextLayer;

                for (int d = 0; d < depth; d++) 
                    for (int nd = 0; nd < next.depth; nd++) 
                        for (int i = 0; i < next.width; i++) 
                            for (int j = 0; j < next.height; j++) 
                                for (int m = 0; m < next.kernelWidth; m++) 
                                    for (int n = 0; n < next.kernelHeight; n++) 
                                        next.matrix.weight[d, nd, m, n] -= currActivated[d, i + m, j + n] * nextError[nd, i, j] * learningRate;

                for (int d = 0; d < depth; d++)
                    for (int nd = 0; nd < next.depth; nd++)
                        for (int i = 0; i < next.width; i++)
                            for (int j = 0; j < next.height; j++)
                                next.matrix.bias[d] -= nextError[nd, i, j] * learningRate;

            }

            return null;
        }
    }

    public class ConvolutionalLayer : Layer2D {

        public ConvolutionalLayer (int depth, int kernelWidth, int kernelHeight) {
            this.depth = depth;
            this.kernelWidth = kernelWidth;
            this.kernelHeight = kernelHeight;
            this.type = Layer.LAYER_2D;
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;
            layer.nextLayer = this;
            matrix = new NeuronMatrix(layer, this);

            Layer2D prev = (Layer2D)prevLayer;
            this.width = prev.width - this.kernelWidth + 1;
            this.height = prev.height - this.kernelHeight + 1;
        }

        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[depth, width, height];

            for (int i = 0; i < depth; i++)
                for (int j = 0; j < width; j++)
                    for (int k = 0; k < height; k++)
                        ret[i, j, k] = matrix.bias[i];

            if (prevLayer is Layer1D) {
                throw new Exception("Invalid Layer");
            } else if (prevLayer is Layer2D) {
                Layer2D prev = (Layer2D)prevLayer;
                for (int pd = 0; pd < prev.depth; pd++)
                    for (int i = 0; i < kernelWidth; i++)
                        for (int j = 0; j < kernelHeight; j++)
                            for (int d = 0; d < depth; d++)
                                for (int m = 0; m < width; m++)
                                    for (int n = 0; n < height; n++)
                                        ret[d, m, n] += prevActivated[pd, i + m, j + n] * matrix.weight[pd, d, i, j];
                            
            }

            for (int i = 0; i < depth; i++)
                for (int j = 0; j < width; j++)
                    for (int k = 0; k < height; k++)
                        ret[i, j, k] = Math.Tanh(ret[i, j, k]);

            return ret;
        }

        public override double[,,] backPropagate (double[,,] currActivated, double[,,] nextError, double learningRate) {
            double[,,] ret = new double[depth, width, height];

            if (nextLayer is Layer1D) {
                Layer1D next = (Layer1D)nextLayer;

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < width; i++)
                        for (int j = 0; j < height; j++)
                            for (int k = 0; k < next.size; k++)
                                ret[d, i, j] += next.matrix.weight[d, i, j, k] * nextError[0, 0, k] * (1 - currActivated[d, i, j] * currActivated[d, i, j]);

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < width; i++)
                        for (int j = 0; j < height; j++)
                            for (int k = 0; k < next.size; k++)
                                next.matrix.weight[d, i, j, k] -= nextError[0, 0, k] * currActivated[d, i, j] * learningRate;

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < next.size; i++)
                        next.matrix.bias[d] -= nextError[0, 0, i] * learningRate;

            } else if (nextLayer is ConvolutionalLayer) {
                ConvolutionalLayer next = (ConvolutionalLayer)nextLayer;

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < next.kernelWidth; i++)
                        for (int j = 0; j < next.kernelHeight; j++)
                            for (int nd = 0; nd < next.depth; nd++)
                                for (int m = 0; m < next.width; m++)
                                    for (int n = 0; n < next.height; n++)
                                        ret[d, i + m, j + n] += next.matrix.weight[d, nd, i, j] * nextError[nd, m, n] * (1 - currActivated[d, i + m, j + n] * currActivated[d, i + m, j + n]);

                for (int d = 0; d < depth; d++)
                    for (int i = 0; i < next.kernelWidth; i++)
                        for (int j = 0; j < next.kernelHeight; j++)
                            for (int nd = 0; nd < next.depth; nd++)
                                for (int m = 0; m < next.width; m++)
                                    for (int n = 0; n < next.height; n++)
                                        next.matrix.weight[d, nd, i, j] -= nextError[nd, m, n] * currActivated[d, i + m, j + n] * learningRate;

                for (int d = 0; d < depth; d++)
                    for (int nd = 0; nd < next.depth; nd++)
                        for (int i = 0; i < next.width; i++)
                            for (int j = 0; j < next.height; j++)
                                next.matrix.bias[d] -= nextError[nd, i, j] * learningRate;

            } else if (nextLayer is MaxPoolingLayer) {
                MaxPoolingLayer next = (MaxPoolingLayer)nextLayer;

                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < next.width; i++) {
                        for (int j = 0; j < next.height; j++) {
                            int maxWidthIndex = i * next.kernelWidth;
                            int maxHeightIndex = j * next.kernelHeight;
                            for (int m = 0; m < next.kernelWidth; m++)
                                for (int n = 0; n < next.kernelHeight; n++)
                                    if (currActivated[d, i * next.kernelWidth + m, j * next.kernelHeight + n] > currActivated[d, maxWidthIndex, maxHeightIndex]) {
                                        maxWidthIndex = i *  next.kernelWidth + m;
                                        maxHeightIndex = j * next.kernelHeight + n;
                                    }
                            ret[d, maxWidthIndex, maxHeightIndex] = nextError[d, i, j];
                        }
                    }
                }
            } else if (nextLayer is MeanPoolingLayer) {
                MeanPoolingLayer next = (MeanPoolingLayer)nextLayer;

                for (int d = 0; d < depth; d++) {
                    for (int i = 0; i < next.width; i++) {
                        for (int j = 0; j < next.height; j++) {
                            double curr = nextError[d, i, j] / next.kernelWidth / next.kernelHeight;
                            for (int m = 0; m < next.kernelWidth; m++)
                                for (int n = 0; n < next.kernelHeight; n++)
                                    ret[d, i * next.kernelWidth + m, j * next.kernelHeight + n] = curr;
                        }
                    }
                }
            }

            return ret;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public abstract class PoolingLayer : Layer2D {
        public PoolingLayer (int kernelWidth, int kernelHeight) {
            this.kernelWidth = kernelWidth;
            this.kernelHeight = kernelHeight;
            this.type = Layer.LAYER_2D;
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;
            layer.nextLayer = this;

            Layer2D prev = (Layer2D)prevLayer;
            this.depth = prev.depth;
            this.width = prev.width / this.kernelWidth;
            this.height = prev.height / this.kernelHeight;
        }
    }

    public class MaxPoolingLayer : PoolingLayer {
        public MaxPoolingLayer (int kernelWidth, int kernelHeight) : base(kernelWidth, kernelHeight) { }

        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[depth, width, height];

            if (prevLayer is Layer1D)
                throw new Exception("Invalid Layer");

            for (int d = 0; d < depth; d++) {
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        int maxWidthIndex = i * kernelWidth;
                        int maxHeightIndex = j * kernelHeight;
                        for (int m = 0; m < kernelWidth; m++)
                            for (int n = 0; n < kernelHeight; n++)
                                if (prevActivated[d, i * kernelWidth + m, j * kernelHeight + n] > prevActivated[d, maxWidthIndex, maxHeightIndex]) {
                                    maxWidthIndex = i * kernelWidth + m;
                                    maxHeightIndex = j * kernelHeight + n;
                                }
                        ret[d, i, j] = prevActivated[d, maxWidthIndex, maxHeightIndex];
                    }
                }
            }

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
            }

            return ret;
        }
    }

    public class MeanPoolingLayer : PoolingLayer {
        public MeanPoolingLayer (int kernelWidth, int kernelHeight) : base(kernelWidth, kernelHeight) { }

        public override double[,,] forwardPropagate (double[,,] prevActivated) {
            double[,,] ret = new double[depth, width, height];

            if (prevLayer is Layer1D)
                throw new Exception("Invalid Layer");

            for (int d = 0; d < depth; d++) {
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        double sum = 0;
                        for (int m = 0; m < kernelWidth; m++) {
                            for (int n = 0; n < kernelHeight; n++) {
                                sum += prevActivated[d, i * kernelWidth + m, j * kernelHeight + n];
                            }
                        }
                        ret[d, i, j] = sum / kernelHeight / kernelWidth;
                    }
                }
            }

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
            }

            return ret;
        }
    }
}

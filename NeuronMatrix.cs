using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public class NeuronMatrix {
        public double[,,,] weight;
        public double[] bias;
        private Random rand = new Random();

        // Three possibilities:
        // 2D -> 2D
        // 2D -> 1D
        // 1D -> 1D
        public NeuronMatrix (Layer l1, Layer l2) {
            Debug.Assert(!(l1.type == Layer.LAYER_1D && l2.type == Layer.LAYER_2D));

            if (l1 is Layer2D && l2 is Layer2D) {
                Layer2D prev = (Layer2D)l1;
                Layer2D curr = (Layer2D)l2;
                weight = new double[prev.depth, curr.depth, curr.kernelWidth, curr.kernelHeight];
                bias = new double[curr.depth];
                initializeWeights(prev.depth * curr.kernelWidth * curr.kernelHeight);
            } else if (l1 is Layer2D && l2 is Layer1D) {
                Layer2D prev = (Layer2D)l1;
                Layer1D curr = (Layer1D)l2;
                weight = new double[prev.depth, prev.width, prev.height, curr.size];
                bias = new double[curr.size];
                initializeWeights(prev.depth * prev.width * prev.height);
            } else if (l1 is Layer1D && l2 is Layer1D) {
                Layer1D prev = (Layer1D)l1;
                Layer1D curr = (Layer1D)l2;
                weight = new double[1, 1, prev.size, curr.size];
                bias = new double[curr.size];
                initializeWeights(prev.size);
            }
        }

        private void initializeWeights (int prevSize) {
            double range = 1 / Math.Sqrt(prevSize);
            for (int i = 0; i < weight.GetLength(0); i++)
                for (int j = 0; j < weight.GetLength(1); j++)
                    for (int k = 0; k < weight.GetLength(2); k++)
                        for (int l = 0; l < weight.GetLength(3); l++)
                            weight[i, j, k, l] = GetRandomGaussian(range);

        }

        private double GetRandomGaussian (double stdDev) {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return stdDev * randStdNormal;
        }
    }
}

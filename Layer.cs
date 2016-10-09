using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public abstract class Layer {
        public const int LAYER_1D = 0;
        public const int LAYER_2D = 1;

        public int type;
        public Layer prevLayer, nextLayer;
        public NeuronMatrix matrix;

        public abstract void BindTo (ref Layer layer);

        public abstract double[,,] forwardPropagate (double[,,] prevActivated);
        public abstract double[,,] backPropagate (double[,,] currActivated, double[,,] nextError, double learningRate);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1 {
    class Program {

        static void Main(string[] args) {

            List<Layer> layers = new List<Layer>();


            layers.Add(new Affine(5,3));
            layers.Add(new Sigmoid());
            layers.Add(new Affine(1,5));
            layers.Add(new Step());
            layers.Add(new Identity());

            List<TrainingSet> trainingSets = new List<TrainingSet>();

            for (int i = 0; i < 1000; i++) {
                trainingSets.Add(new TrainingSet(new double[3] { 0, 0, 0 }, new double[1] { 0 }));
                trainingSets.Add(new TrainingSet(new double[3] { 0, 0, 1 }, new double[1] { 1 }));
                trainingSets.Add(new TrainingSet(new double[3] { 0, 1, 0 }, new double[1] { 1 }));
                trainingSets.Add(new TrainingSet(new double[3] { 0, 1, 1 }, new double[1] { 0 }));
                trainingSets.Add(new TrainingSet(new double[3] { 1, 0, 0 }, new double[1] { 1 }));
                trainingSets.Add(new TrainingSet(new double[3] { 1, 0, 1 }, new double[1] { 0 }));
                trainingSets.Add(new TrainingSet(new double[3] { 1, 1, 0 }, new double[1] { 0 }));
                trainingSets.Add(new TrainingSet(new double[3] { 1, 1, 1 }, new double[1] { 1 }));
            }


            
            for (int i = 0; i < trainingSets.Count(); i++) {
                double[] data = (double[])trainingSets[i].Features.Clone();
                for (int j = 0; j < layers.Count(); j++) {
                    data = layers[j].ForwardPropagation(data);
                }
                double[] error = (double[]) trainingSets[i].Target.Clone();
                for (int j = layers.Count()-1; j >= 0; j--) {
                    error = layers[j].BackPropagation(error);
                }
                Console.Write(String.Format("Expected : {0}   result: {1}   ", trainingSets[i].Target[0], data[0]));
                if (Math.Abs(data[0] - trainingSets[i].Target[0]) < 0.1) Console.WriteLine("v");
                else Console.WriteLine("X");
            }
            

        }
    }


    class TrainingSet {
        public double[] Features { get; private set; }
        public double[] Target { get; private set; }

        public TrainingSet(double[] features, double[] target) {
            Features = features;
            Target = target;
        }

    }



    abstract class Layer {
        abstract public double[] ForwardPropagation(double[] input);
        abstract public double[] BackPropagation(double[] input);
    }



    class Affine : Layer{
        static Random random = new Random();

        private double[] value;
        private double[,] weight;
        private double[] bias;

        public Affine(int nodes, int inputNodes){
            weight = new double[nodes,inputNodes];
            bias = new double[nodes];
            for (int i = 0; i < weight.GetLength(0); i++) {
                for (int j = 0; j < weight.GetLength(1); j++) {
                    weight[i,j] = (Math.Sqrt(-2.0 * Math.Log(random.NextDouble())) * Math.Sin(2.0 * Math.PI * random.NextDouble())) * (1.0 / Math.Sqrt(weight.GetLength(1)));
                }
            }
        }

        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[weight.GetLength(0)];
            for (int i = 0; i < weight.GetLength(0); i++) {
                double sum = 0;
                for (int j = 0; j < weight.GetLength(1); j++) {
                    sum += (input[j] * weight[i,j]);
                }
                output[i] = sum + bias[i];
            }
            value = (double[])input.Clone();
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[weight.GetLength(1)];
            for (int j = 0; j < weight.GetLength(1); j++) {
                for (int i = 0; i < weight.GetLength(0); i++) {
                    output[j] += (input[i] * weight[i,j]);
                }
            }

            for (int j = 0; j < weight.GetLength(1); j++) {
                for (int i = 0; i < weight.GetLength(0); i++) {
                    weight[i, j] -= input[i] * value[j] * 0.5;
                }
            }
            for (int i = 0; i < weight.GetLength(0); i++) {
                bias[i] -= input[i] * 0.5;
            }

            return output;
        }
    }




    class Sigmoid : Layer {
        private double[] value;

        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
            }
            value = (double[])output.Clone();
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                output[i] = input[i] * ((1.0 - value[i]) * value[i]);
            }
            return output;
        }
    }




    class ReLU : Layer {
        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                if (input[i] >= 0) {
                    output[i] = input[i];
                } else {
                    output[i] = 0;
                }
            }
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                if (input[i] >= 0) {
                    output[i] = 1;
                } else {
                    output[i] = 0;
                }
            }
            return output;
        }
    }





    class Step : Layer {
        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                if (input[i] >= 0) {
                    output[i] = 1;
                } else {
                    output[i] = 0;
                }
            }
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = (double[])input.Clone();
            return output;
        }
    }





    class Softmax : Layer {
        private double[] value;

        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            double sum = input.Sum();
            double max = input.Max();
            for (int i = 0; i < input.Length; i++) {
                output[i] = Math.Exp(input[i] - max)/Math.Exp(sum - max);
            }
            value = (double[])input.Clone();
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                output[i] = value[i] - input[i];
            }
            return output;
        }
    }



    class Identity : Layer {
        private double[] value;

        override public double[] ForwardPropagation(double[] input) {
            double[] output = (double[])input.Clone();
            value = (double[])input.Clone();
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                output[i] = value[i] - input[i];
            }
            return output;
        }
    }


}

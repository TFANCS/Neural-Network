using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace NeuralNetwork1 {
    class Program {

        static void Main(string[] args) {

            List<Layer> layers = new List<Layer>();

            layers.Add(new Affine(100,784));
            layers.Add(new ReLU());
            layers.Add(new Affine(10, 100));
            layers.Add(new Softmax());

            List<TrainingSet> trainingSets = new List<TrainingSet>();
          

            for (int i = 0; i < 10; i++) {
                IEnumerable<string> files = System.IO.Directory.EnumerateFiles(@"D:\Desktop\mnist_png\training\" + i, "*", System.IO.SearchOption.AllDirectories);
                Parallel.ForEach(files, file => {
                    TrainingSet temp = new TrainingSet(BitmapToArray.Convert(new Bitmap(file)), new double[10].Select((x, index) => index == i ? 1.0 : 0.0).ToArray());
                    lock (trainingSets) {
                        trainingSets.Add(temp);
                    }
                });
            }


            //shuffle training sets
            trainingSets = trainingSets.OrderBy(i => Guid.NewGuid()).ToList();





            foreach (TrainingSet item in trainingSets) {
                double[] data = (double[])item.Features.Clone();
                for (int j = 0; j < layers.Count(); j++) {
                    data = layers[j].ForwardPropagation(data);
                }
                double[] error = (double[])item.Target.Clone();
                for (int j = layers.Count() - 1; j >= 0; j--) {
                    error = layers[j].BackPropagation(error);
                }
                
                //double a = trainingSets[i].Target[0];
                //double b = data[0];
                int a = Array.IndexOf(item.Target, item.Target.Max());
                int b = Array.IndexOf(data, data.Max());
                Console.Write(String.Format("Expected : {0}   result: {1}   ", a, b));
                if (Math.Abs(a - b) < 0.1) Console.WriteLine("v");
                else Console.WriteLine("X");
                
            }

            Console.WriteLine("finished");


            while (true) {
                try {
                    double[] data = BitmapToArray.Convert(new Bitmap(Console.ReadLine()));
                    for (int j = 0; j < layers.Count(); j++) {
                        data = layers[j].ForwardPropagation(data);
                    }
                    Console.WriteLine(Array.IndexOf(data, data.Max()));
                } catch {
                }
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
        private double[,] diffWeight;
        private double[] diffBias;

        private double learningRate = 0.0001;
        private Optimizer optimizer;

        public Affine(int nodes, int inputNodes){
            weight = new double[nodes,inputNodes];
            diffWeight = new double[nodes, inputNodes];
            bias = new double[nodes];
            diffBias = new double[nodes];
            optimizer = new SGD();
            //optimizer = new Momentum(nodes,inputNodes);
            for (int i = 0; i < weight.GetLength(0); i++) {
                for (int j = 0; j < weight.GetLength(1); j++) {
                    weight[i,j] = (Math.Sqrt(-2.0 * Math.Log(random.NextDouble())) * Math.Sin(2.0 * Math.PI * random.NextDouble())) * Math.Sqrt(2.0 / weight.GetLength(1));
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
            value = input;
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
                    diffWeight[i,j] = input[i] * value[j] * learningRate;
                }
            }
            for (int i = 0; i < weight.GetLength(0); i++) {
                diffBias[i] = input[i] * learningRate;
            }

            optimizer.Update(ref weight, ref diffWeight);
            optimizer.Update(ref bias, ref bias);

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
            value = output;
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




    class Tanh : Layer {
        private double[] value;

        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                output[i] = (Math.Exp(input[i]) - Math.Exp(-input[i])) / (Math.Exp(input[i]) + Math.Exp(-input[i]));
            }
            value = output;
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                output[i] = input[i] * (1.0 - (value[i] * value[i]));
            }
            return output;
        }
    }





    class ReLU : Layer {
        private double[] value;

        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                if (input[i] > 0) {
                    output[i] = input[i];
                } else {
                    output[i] = 0;
                }
            }
            value = input;
            return output;
        }

        override public double[] BackPropagation(double[] input) {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++) {
                if (value[i] > 0) {
                    output[i] = input[i];
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
            double[] output = input;
            return output;
        }
    }





    class Softmax : Layer {
        private double[] value;

        override public double[] ForwardPropagation(double[] input) {
            double[] output = new double[input.Length];
            double max = input.Max();
            double sum = input.Select(x => Math.Exp(x - max)).Sum();
            for (int i = 0; i < input.Length; i++) {
                output[i] = Math.Exp(input[i] - max) / sum;
            }
            value = output;
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
            double[] output = input;
            value = input;
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






    abstract class Optimizer {
        abstract public void Update(ref double[] input, ref double[] diff);
        abstract public void Update(ref double[,] input, ref double[,] diff);
    }


    class SGD : Optimizer {
        override public void Update(ref double[] input, ref double[] diff) {
            for (int i = 0; i < input.Length; i++) {
                input[i] -= diff[i];
            }
        }

        override public void Update(ref double[,] input, ref double[,] diff) {
            for (int i = 0; i < input.GetLength(0); i++) {
                for (int j = 0; j < input.GetLength(1); j++) {
                    input[i, j] -= diff[i, j];
                }
            }

        }
    }





    class Momentum : Optimizer {
        private double[] velocity1;
        private double[,] velocity2;

        public Momentum(int nodes, int inputNodes) {
            velocity1 = new double[nodes];
            velocity2 = new double[nodes, inputNodes];
        }

        override public void Update(ref double[] input, ref double[] diff) {
            for (int i = 0; i < input.Length; i++) {
                velocity1[i] = (velocity1[i] * 0.1) - diff[i];
                input[i] += velocity1[i];
            }
        }

        override public void Update(ref double[,] input, ref double[,] diff) {
            for (int i = 0; i < input.GetLength(0); i++) {
                for (int j = 0; j < input.GetLength(1); j++) {
                    velocity2[i, j] = (velocity2[i, j] * 0.1) - diff[i, j];
                    input[i, j] += velocity2[i, j];
                }
            }

        }
    }









    static class BitmapToArray {
        static public double[] Convert(Bitmap bmp) {
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);

            IntPtr ptr = bmpData.Scan0;

            int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
            byte[] rgbValues = new byte[bytes];

            Marshal.Copy(ptr, rgbValues, 0, bytes);

            bmp.UnlockBits(bmpData);

            bmp.Dispose();

            double[] output = rgbValues.Select(x => (double)x).Where((x, index)=> index%4==0).ToArray();

            return output;
        }

    }



}

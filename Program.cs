using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1 {
    class Program {

        static void Main(string[] args) {

            Neuron neuronIn1 = new Neuron(1);
            Neuron neuronIn2 = new Neuron(1);
            Neuron neuronMid1 = new Neuron(1);
            Neuron neuronMid2 = new Neuron(1);
            Neuron neuronOut = new Neuron(0);
            neuronMid1.Connect(new Neuron[2] { neuronIn1, neuronIn2 });
            neuronMid2.Connect(new Neuron[2] { neuronIn1, neuronIn2 });
            neuronOut.Connect(new Neuron[2] { neuronMid1, neuronMid2 });

            List<TrainingSet> trainingSets = new List<TrainingSet>();

            for (int i = 0; i < 400; i++) {
                trainingSets.Add(new TrainingSet(new double[2] { 0, 0 }, 0));
                trainingSets.Add(new TrainingSet(new double[2] { 0, 1 }, 1));
                trainingSets.Add(new TrainingSet(new double[2] { 1, 0 }, 1));
                trainingSets.Add(new TrainingSet(new double[2] { 1, 1 }, 0));
            }


            neuronMid1.SetWeight(1, 1);
            neuronMid2.SetWeight(1, 0);
            neuronOut.SetBias(1);


            for (int i = 0; i < trainingSets.Count(); i++) {
                neuronIn1.SetValue(trainingSets[i].Features[0]);
                neuronIn2.SetValue(trainingSets[i].Features[1]);
                neuronMid1.ForwardPropagation();
                neuronMid2.ForwardPropagation();
                neuronOut.ForwardPropagation();
                neuronOut.CalcError(trainingSets[i].Target);
                neuronOut.BackPropagation();
                double result = neuronOut.GetValue();
                Console.Write("Expectd : "+ trainingSets[i].Target + "   result: " + result + "   ");
                if (Math.Abs(result - trainingSets[i].Target) < 0.1) Console.WriteLine("v");
                else Console.WriteLine("X");
            }



        }
    }


    class TrainingSet {
        public double[] Features { get; private set; }
        public double Target { get; private set; }

        public TrainingSet(double[] features, double target) {
            Features = features;
            Target = target;
        }

    }




    class Neuron {
        private double[] weight = new double[2];
        private double bias = 0;
        private readonly double learningRate = 0.5;
        private double value = 0;    //sum of weighted imputs and bias
        private double error = 0;    //difference between target and value

        private readonly int actFunc;  //0->step  1->sigmoid  2->ReLU

        private Neuron[] input;

        public Neuron(int actFunc) {
            this.actFunc = actFunc;
        }

        public void Connect(Neuron[] input) {
            this.input = input;
            return;
        }

        public void SetValue(double value) {
            this.value = value;
        }

        public double GetValue() {
            return value;
        }

        public void SetWeight(double weight, int index) {
            this.weight[index] = weight;
            return;
        }

        public void SetBias(double bias) {
            this.bias = bias;
            return;
        }


        public void ForwardPropagation() {
            int length = input.Length;

            double sum = 0;
            for (int i = 0; i < length; i++) {
                sum += input[i].GetValue() * weight[i];
            }
            sum += bias;
            double result = 0;
            switch (actFunc) {
                case 0:
                    result = Step(sum);
                    break;
                case 1:
                    result = Sigmoid(sum);
                    break;
                case 2:
                    result = ReLU(sum);
                    break;
            }
            value = result;
            
            return;
        }



        public void BackPropagation() {
            int length = input.Length;

            for (int i = 0; i < length; i++) {
                input[i].UpdateError(error, weight[i]);
                input[i].UpdateWeight();
            }

            return;
        }



        public void CalcError(double target) {
            error = (target - value);
            UpdateWeight();
        }


        public void UpdateError(double errorBack, double weightBack) {
            int length = input.Length;

            switch (actFunc) {
                case 0:
                    error = (errorBack * weightBack);
                    break;
                case 1:
                    error = (errorBack * weightBack * SigmoidDeriv(value));
                    break;
                case 2:
                    error = (errorBack * weightBack);
                    break;
            }
            return;
        }



        public void UpdateWeight() {
            int length = input.Length;

            switch (actFunc) {
                case 0:
                    for (int i = 0; i < length; i++) {
                        weight[i] = weight[i] + (learningRate * error * input[i].GetValue());
                    }
                    bias = bias + (learningRate * error);
                    break;
                case 1:
                    for (int i = 0; i < length; i++) {
                        weight[i] = weight[i] + (learningRate * error * SigmoidDeriv(value) * input[i].GetValue());
                    }
                    bias = bias + (learningRate * error * SigmoidDeriv(value));
                    break;
                case 2:
                    for (int i = 0; i < length; i++) {
                        weight[i] = weight[i] + (learningRate * error * input[i].GetValue());
                    }
                    bias = bias + (learningRate * error);
                    break;
            }
        }



        private double ReLU(double x) {
            if (x >= 0) {
                return x;
            }else {
                return 0;
            }
        }

        private double Sigmoid(double x) {
            return 1 / (1+Math.Exp(-x));
        }


        private double SigmoidDeriv(double y) {
            return y*(1-y);
        }


        private double Step(double x) {
            if (x >= 0) {
                return 1;
            } else {
                return 0;
            }
        }


    }




}

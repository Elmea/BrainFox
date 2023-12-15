using System.Diagnostics;

namespace BrainFoxCS
{
    enum ActivationFunction 
    {
        threshold,
        sigmoid,
        tanh,
        ReLU
    }

    namespace Component
    {
        abstract class Perceptron
        {
            public float error = 0;
            protected float output = 0;
            public float Output { get { return output; } }

            // Only used with sigmoid activation function
            public float sigmoidBeta = 1; 
            public ActivationFunction activationFunction = ActivationFunction.ReLU;

            abstract public void CalcOutput();

            #region ActivationFunction

            protected float CalcActivationFunction(float input)
            {
                switch (activationFunction)
                {
                    case ActivationFunction.threshold:
                        return ThresholdFunction(input);
                
                    case ActivationFunction.sigmoid: 
                        return SigmoidFunction(input, sigmoidBeta);

                    case ActivationFunction.tanh:
                        return TanhFunction(input);
                 
                    case ActivationFunction.ReLU: 
                        return ReLUFunction(input);

                    default:
                        Debug.Assert(false, "WARNING : Activation function not properly setted or not implemented");
                        return 0;
                }
            }

            protected static float ThresholdFunction(float input)
            {
                if (input < 0) return 0;

                return 1;
            }

            protected static float SigmoidFunction(float input, float beta) 
            {
                return  1 / (1 + MathF.Exp(-(beta - input)));
            }

            protected static float TanhFunction(float input) 
            {
                return MathF.Tanh(input);
            }

            protected static float ReLUFunction(float input) 
            {
                if (input < 0) return 0;

                return input;
            }

            #endregion
        }

        class InnerPerceptron : Perceptron
        {
            protected class WeightedInputPerceptron
            {
                public Perceptron perceptron;
                public float weight;
            }

            private List<WeightedInputPerceptron> inputs = new List<WeightedInputPerceptron>(); // All input Perceptron and their weight value

            public float[] GetConnectionWeights()
            {
                float[] weights = new float[inputs.Count];

                for (int i = 0; i < inputs.Count; i++)
                {
                    weights[i] = inputs[i].weight;
                }

                return weights;
            }

            override public void CalcOutput() 
            {
                float input = 0;

                foreach (WeightedInputPerceptron weightedInput in inputs) 
                {
                    input = weightedInput.perceptron.Output * weightedInput.weight;
                }

                output = CalcActivationFunction(input);
            }

            public void AddInput(Perceptron perceptron)
            {
                WeightedInputPerceptron newOne = new WeightedInputPerceptron();
                newOne.perceptron = perceptron;
                newOne.weight = 1;

                inputs.Add(newOne);
            }

            public void RemoveInput(Perceptron perceptron)
            {
                foreach (WeightedInputPerceptron input in inputs)
                {
                    if (input.perceptron == perceptron)
                    {
                        inputs.Remove(input);
                        return;
                    }
                }
            }

            public void ResetConnections()
            { 
                inputs.Clear(); 
            }

            public void Backpropagation(float gain)
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    float delta = gain * error * inputs[i].perceptron.Output;
                    inputs[i].weight += delta;
                }
            }

            public float GetIncomingWeight(int index)
            {
                try
                {
                    return inputs[index].weight;
                }
                catch (IndexOutOfRangeException e)
                {
                    throw new ArgumentOutOfRangeException("Index is out of range : ", e);
                }
            }
        }

        class InputPerceptron : Perceptron
        {
            public float input = 0;

            override public void CalcOutput()
            {
                output = CalcActivationFunction(input);
            }
        }
    }
}

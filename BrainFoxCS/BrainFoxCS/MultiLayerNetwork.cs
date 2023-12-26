using BrainFoxCS.Component;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace BrainFoxCS
{
    [Serializable()]
    class MultiLayerNetwork
    {
        public Layer.InputLayer inputLayer;
        private List<Layer.InnerLayer> hiddenLayers;
        private Layer.InnerLayer outputLayer;

        /// <summary>
        /// Set a name for the file, .brfox will be added
        /// </summary>
        static public MultiLayerNetwork SaveNetwork(string fileName)
        {
            //Format the object as Binary  
            BinaryFormatter formatter = new BinaryFormatter();

            //Reading the file from the server  
            FileStream fs = File.Open(fileName, FileMode.Open);

            object obj = formatter.Deserialize(fs);

            MultiLayerNetwork result = (MultiLayerNetwork)obj;
            fs.Close();

            return result;
        }

        /// <summary>
        /// Look for a file with the .brfox  expansion
        /// </summary>
        public void LoadNetwork(string fileName)
        {
            //Format the object as Binary  
            BinaryFormatter formatter = new BinaryFormatter();

            //Reading the file from the server  
            FileStream fs = File.OpenWrite(fileName);

            formatter.Serialize(fs, this);
            fs.Close();
        }

        public int GetLayerCount()
        {
            return hiddenLayers.Count + 2;
        }

        public int GetHiddenLayerCount()
        {
            return hiddenLayers.Count;
        }

        public int[] GetPerceptronsByLayer()
        {
            int[] perceptronsByLayer = new int[GetLayerCount()];

            perceptronsByLayer[0] = inputLayer.GetPerceptronCount();
            perceptronsByLayer[GetLayerCount() - 1] = outputLayer.GetPerceptronCount();

            for (int i = 0; i < hiddenLayers.Count; i++)
            {
                perceptronsByLayer[i + 1] = hiddenLayers[i].GetPerceptronCount();
            }

            return perceptronsByLayer;
        }

        public List<float[]> GetPercepWeightsOfLayer(int index)
        {
            if (index == hiddenLayers.Count)
                return outputLayer.GetPercepWeights();

            return hiddenLayers[index].GetPercepWeights();
        }

        #region OutputLayer
        public void AddOutput()
        {
            outputLayer.CreateOutput();
        }

        public void RemoveOutput()
        {
            outputLayer.RemoveOutput();
        }

        public List<float> GetOutputValues()
        {
            return outputLayer.GetOutputValues();
        }
        #endregion

        /// <summary>
        /// Calc the outputs of the neuronal network and return it.
        /// </summary>
        public List<float> CalcOutputs()
        {
            inputLayer.CalcOutputs();
            foreach (Layer.InnerLayer layer in hiddenLayers)
                layer.CalcOutputs();
            outputLayer.CalcOutputs();
            return outputLayer.GetOutputValues();
        }

        public void SetAllFunction(ActivationFunction function)
        {
            outputLayer.SetActivationFunction(function);
            foreach (Layer.InnerLayer layer in hiddenLayers)
                layer.SetActivationFunction(function);
        }

        public void SetHiddenLayerWeights(int index, List<float[]> weight)
        {
            hiddenLayers[index].SetPercepWeights(weight);
        }


        public void SetHiddenLayerFunction(int index, ActivationFunction function)
        {
            hiddenLayers[index].SetActivationFunction(function);
        }

        public void SetOutputLayerFunction(ActivationFunction function)
        {
            outputLayer.SetActivationFunction(function);
        }

        #region HiddenLayersFunction

        public void CreateHiddenLayer(int perceptronCount = 1)
        {
            if (perceptronCount < 1)
                return;

            bool firstOne = hiddenLayers.Count == 0;
            Layer.InnerLayer newOne = new Layer.InnerLayer(perceptronCount);

            if (firstOne)
            {
                hiddenLayers.Add(newOne);
                hiddenLayers[0].BindLeadingLayer(inputLayer);
                outputLayer.BindLeadingLayer(hiddenLayers[0]);
            }
            else
            {
                hiddenLayers.Add(newOne);

                int newIndex = hiddenLayers.Count - 1;

                hiddenLayers[newIndex].BindLeadingLayer(hiddenLayers[newIndex - 1]);
                outputLayer.BindLeadingLayer(hiddenLayers[newIndex]);
            }
        }

        public void RemoveHiddenLayer(int index)
        {
            try
            { 
                bool firstOne = index ==  0;
                bool lastOne = index == hiddenLayers.Count - 1;
                bool onlyOneLeft = hiddenLayers.Count == 0;

                if (onlyOneLeft)
                {
                    hiddenLayers.Clear();
                    outputLayer.BindLeadingLayer(inputLayer);
                    return;
                }

                hiddenLayers.RemoveAt(index);
                if (firstOne)
                {
                    hiddenLayers[0].BindLeadingLayer(inputLayer);
                }
                else if (lastOne)
                {
                    if (hiddenLayers.Count == 1)
                        hiddenLayers[hiddenLayers.Count - 1].BindLeadingLayer(inputLayer);
                    else
                        hiddenLayers[hiddenLayers.Count - 1].BindLeadingLayer(hiddenLayers[index - 1]);
                }
                else
                {
                    hiddenLayers[index].BindLeadingLayer(hiddenLayers[index - 1]);
                }
            }
            catch (IndexOutOfRangeException e)
            {
                throw new ArgumentOutOfRangeException("Index is out of range : ", e);
            }
        }

        public void AddPerceptronToHIddenLayer(int layerIndex)
        {
            try
            {
                hiddenLayers[layerIndex].CreateOutput();
            }
            catch (IndexOutOfRangeException e)
            {
                throw new ArgumentOutOfRangeException("Index is out of range : ", e);
            }
        }

        public void RemovePerceptronToHIddenLayer(int layerIndex)
        {
            try
            {
                hiddenLayers[layerIndex].RemoveOutput();
            }
            catch (IndexOutOfRangeException e)
            {
                throw new ArgumentOutOfRangeException("Index is out of range : ", e);
            }
        }

        #endregion

        public MultiLayerNetwork(int inputsCount = 1, int outputsCount = 1) 
        {
            if (inputsCount < 1)
                inputsCount = 1;

            if (outputsCount < 1)
                outputsCount = 1;

            hiddenLayers = new List<Layer.InnerLayer>();
            outputLayer = new Layer.InnerLayer(outputsCount);
            inputLayer = new Layer.InputLayer(inputsCount, outputLayer);
        }

        public void BackPropagation(float gain, float[] desiredOutput)
        {
            CalcOutputs();

            outputLayer.OutputBackPropagation(gain, desiredOutput);
            for (int i = hiddenLayers.Count-1; i >= 0; i--)
            {
                hiddenLayers[i].HiddenBackPropagation(gain);
            }
        }
    }
}

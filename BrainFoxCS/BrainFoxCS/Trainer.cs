using System.Collections.Generic;
using System.Diagnostics;
using static BrainFoxCS.Trainer;

namespace BrainFoxCS
{
    static class Trainer
    {
        public class TrainingTable
        {
            public float[][] referenceInputs;
            public float[][] referenceOutputs;

            public TrainingTable(int trainCount, int inputCount, int outPutCount)
            {
                referenceInputs = new float[trainCount][];
                referenceOutputs = new float[trainCount][];
                for (int i = 0; i < trainCount; i++)
                {
                    referenceInputs[i] = new float[inputCount];
                    referenceOutputs[i] = new float[outPutCount];
                }
            }
        }

        static public void TrainByBackPropagation(MultiLayerNetwork networkToTrain, 
                                    TrainingTable trainingTable, int iteration, float gain)
        {
            if (trainingTable.referenceInputs.Length != trainingTable.referenceOutputs.Length) 
            {
                if (trainingTable.referenceInputs.Length > trainingTable.referenceOutputs.Length)
                {
                    Debug.Assert(false, "!!! Reference outputs are missing !!!");
                    return;
                }

                Debug.Assert(false, "!!! Reference inputs are missing !!!");
            }

            for (int i = 0; i < iteration; i++) 
            {
                for (int j = 0; j < trainingTable.referenceInputs.Length; j++) 
                {
                    networkToTrain.inputLayer.SetInputs(trainingTable.referenceInputs[j]);
                    networkToTrain.BackPropagation(gain, trainingTable.referenceOutputs[j]);
                }
            }
        }
    }
}

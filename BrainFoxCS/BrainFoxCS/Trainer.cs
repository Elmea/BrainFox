using System.Diagnostics;

namespace BrainFoxCS
{
    static class Trainer
    {
        public struct TrainingTable
        {
            public List<float[]> referenceInputs;
            public List<float[]> referenceOutputs;
        }

        static public void TrainByBackPropagation(MultiLayerNetwork networkToTrain, 
                                    TrainingTable trainingTable, int iteration, float gain)
        {
            if (trainingTable.referenceInputs.Count != trainingTable.referenceOutputs.Count) 
            {
                if (trainingTable.referenceInputs.Count > trainingTable.referenceOutputs.Count)
                {
                    Debug.Assert(false, "!!! Reference outputs are missing !!!");
                    return;
                }

                Debug.Assert(false, "!!! Reference inputs are missing !!!");
            }

            for (int i = 0; i < iteration; i++) 
            {
                for (int j = 0; j < trainingTable.referenceInputs.Count; j++) 
                {
                    networkToTrain.inputLayer.SetInputs(trainingTable.referenceInputs[j]);
                    networkToTrain.BackPropagation(gain, trainingTable.referenceOutputs[j]);
                }
            }
        }
    }
}

# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
#


# Perceptron implementation
import util

PRINT = True


class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys()  # Could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i, data in enumerate(trainingData):
                scores = util.Counter()
                for label in self.legalLabels:
                    scores[label] = self.weights[label] * data  # Dot product between weights and features
                predictedLabel = scores.argMax()
                trueLabel = trainingLabels[i]

                if predictedLabel != trueLabel:
                    self.weights[trueLabel] += data
                    self.weights[predictedLabel] -= data

        print("Finished training")

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.Counter from features to values.
        """
        guesses = []
        for datum in data:
            vect = util.Counter()
            for label in self.legalLabels:
                vect[label] = self.weights[label] * datum  # Dot product
            guesses.append(vect.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label.
        """
        return self.weights[label].sortedKeys()[:100]

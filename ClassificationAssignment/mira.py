# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util

PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys()  # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        bestAccuracyCount = -1
        bestWeights = {}

        for c in cGrid:
            self.weights = {label: util.Counter() for label in self.legalLabels}
            for iteration in range(self.max_iterations):
                for i in range(len(trainingData)):
                    currentFeatures = trainingData[i]
                    trueLabel = trainingLabels[i]
                    scores = util.Counter()

                    for label in self.legalLabels:
                        scores[label] = currentFeatures * self.weights[label]

                    predictedLabel = scores.argMax()

                    if predictedLabel != trueLabel:
                        f = currentFeatures
                        tau = ((self.weights[predictedLabel] - self.weights[trueLabel]) * f + 1.0) / (2.0 * (f * f))
                        tau = min(c, tau)

                        f_scaled = f.copy()
                        f_scaled.divideAll(1.0 / tau)
                        self.weights[trueLabel] += f_scaled
                        self.weights[predictedLabel] -= f_scaled

            validationGuesses = self.classify(validationData)
            correct = [validationGuesses[i] == validationLabels[i] for i in range(len(validationLabels))]
            accuracy = correct.count(True) / len(correct)

            if accuracy > bestAccuracyCount:
                bestAccuracyCount = accuracy
                bestWeights = self.weights.copy()
                bestParameter = c

        self.weights = bestWeights
        print("Finished training. Best C parameter =", bestParameter)
        return bestAccuracyCount

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        "*** YOUR CODE HERE ***"
        for datum in data:
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = self.weights[label] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        "*** YOUR CODE HERE ***"
        featureWeights = list(self.weights[label].items())
        featureWeights.sort(key=lambda feature: feature[1], reverse=True)
        topFeatures = [feature for feature, weight in featureWeights[:100]]
        return topFeatures

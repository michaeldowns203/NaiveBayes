import java.util.*;

public class GaussianNaiveBayesClassifier {

    private Map<Object, Integer> classCounts;
    private Map<Object, Map<Integer, List<Double>>> featureSums;
    private Map<Object, Map<Integer, List<Double>>> featureSquares;
    private Map<Object, Double> priorProbabilities;
    private Map<Object, Map<Integer, Double>> means;
    private Map<Object, Map<Integer, Double>> variances;
    private int totalExamples;
    private int numAttributes;

    public GaussianNaiveBayesClassifier(int numAttributes) {
        this.classCounts = new HashMap<>();
        this.featureSums = new HashMap<>();
        this.featureSquares = new HashMap<>();
        this.priorProbabilities = new HashMap<>();
        this.means = new HashMap<>();
        this.variances = new HashMap<>();
        this.totalExamples = 0;
        this.numAttributes = numAttributes;
    }

    public void train(Double[][] data, Object[] labels) {
        totalExamples = data.length;

        // Initialize maps for mean and variance calculations
        for (int i = 0; i < data.length; i++) {
            Object label = labels[i];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);

            for (int j = 0; j < data[i].length; j++) {
                double value = data[i][j];

                // Sum and sum of squares for mean/variance calculation
                featureSums.computeIfAbsent(label, k -> new HashMap<>())
                        .computeIfAbsent(j, k -> new ArrayList<>(Arrays.asList(0.0, 0.0)))
                        .set(0, featureSums.get(label).get(j).get(0) + value);

                featureSquares.computeIfAbsent(label, k -> new HashMap<>())
                        .computeIfAbsent(j, k -> new ArrayList<>(Arrays.asList(0.0, 0.0)))
                        .set(0, featureSquares.get(label).get(j).get(0) + value * value);
            }
        }

        // Calculate prior probabilities Q(C = ci)
        for (Object label : classCounts.keySet()) {
            priorProbabilities.put(label, (double) classCounts.get(label) / totalExamples);
        }

        // Calculate means and variances for each feature per class
        for (Object label : classCounts.keySet()) {
            int classCount = classCounts.get(label);
            means.put(label, new HashMap<>());
            variances.put(label, new HashMap<>());

            for (int j = 0; j < numAttributes; j++) {
                double sum = featureSums.get(label).get(j).get(0);
                double sumSquares = featureSquares.get(label).get(j).get(0);

                double mean = sum / classCount;
                double variance = (sumSquares / classCount) - (mean * mean); // Variance formula

                means.get(label).put(j, mean);
                variances.get(label).put(j, variance > 0 ? variance : 1e-6); // Prevent division by zero
            }
        }
    }

    public Object classify(Double[] instance) {
        Map<Object, Double> classScores = new HashMap<>();

        for (Object label : priorProbabilities.keySet()) {
            double score = Math.log(priorProbabilities.get(label)); // Log to prevent underflow

            for (int j = 0; j < instance.length; j++) {
                double mean = means.get(label).get(j);
                double variance = variances.get(label).get(j);
                double value = instance[j];

                // Gaussian PDF
                double probability = (1 / Math.sqrt(2 * Math.PI * variance)) *
                        Math.exp(-((value - mean) * (value - mean)) / (2 * variance));

                score += Math.log(probability); // Log probabilities to prevent underflow
            }

            classScores.put(label, score);
        }

        return classScores.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }
}
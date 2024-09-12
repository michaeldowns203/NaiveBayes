import java.util.*;

public class NaiveBayesClassifier {

    private Map<Object, Integer> classCounts;
    private Map<Object, Map<Integer, Map<Object, Integer>>> featureCounts;
    private Map<Object, Double> priorProbabilities;
    private Map<Object, Map<Integer, Map<Object, Double>>> likelihoods;
    private int totalExamples;
    private int numAttributes;

    public NaiveBayesClassifier(int numAttributes) {
        this.classCounts = new HashMap<>();
        this.featureCounts = new HashMap<>();
        this.priorProbabilities = new HashMap<>();
        this.likelihoods = new HashMap<>();
        this.totalExamples = 0;
        this.numAttributes = numAttributes;
    }

    public void train(Object[][] data, Object[] labels) {
        totalExamples = data.length;

        // Count occurrences of each class and each feature value per class
        for (int i = 0; i < data.length; i++) {
            Object label = labels[i];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);

            for (int j = 0; j < data[i].length; j++) {
                int attribute = j;
                Object value = data[i][j];
                featureCounts.computeIfAbsent(label, k -> new HashMap<>())
                        .computeIfAbsent(attribute, k -> new HashMap<>())
                        .put(value, featureCounts.get(label).get(attribute).getOrDefault(value, 0) + 1);
            }
        }

        // Calculate prior probabilities Q(C = ci)
        for (Object label : classCounts.keySet()) {
            priorProbabilities.put(label, (double) classCounts.get(label) / totalExamples);
        }

        // Calculate likelihoods F(Aj = ak, C = ci)
        for (Object label : classCounts.keySet()) {
            int classCount = classCounts.get(label);
            likelihoods.put(label, new HashMap<>());

            for (int attribute : featureCounts.get(label).keySet()) {
                likelihoods.get(label).put(attribute, new HashMap<>());
                for (Object value : featureCounts.get(label).get(attribute).keySet()) {
                    int count = featureCounts.get(label).get(attribute).get(value);
                    double likelihood = (double) (count + 1) / (classCount + numAttributes); // Smoothing
                    likelihoods.get(label).get(attribute).put(value, likelihood);
                }
            }
        }
    }

    public Object classify(Object[] instance) {
        Map<Object, Double> classScores = new HashMap<>();

        for (Object label : priorProbabilities.keySet()) {
            double score = priorProbabilities.get(label);
            for (int j = 0; j < instance.length; j++) {
                int attribute = j;
                Object value = instance[j];
                score *= likelihoods.get(label).get(attribute).getOrDefault(value, 1.0 / (classCounts.get(label) + numAttributes));
            }
            classScores.put(label, score);
        }

        return classScores.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    // Method to print the trained model
    public void printModel() {
        System.out.println("Prior Probabilities:");
        for (Object label : priorProbabilities.keySet()) {
            System.out.println("Class " + label + ": " + priorProbabilities.get(label));
        }

        System.out.println("\nClass-Conditional Attribute Probabilities:");
        for (Object label : likelihoods.keySet()) {
            System.out.println("Class " + label + ":");
            for (int attribute : likelihoods.get(label).keySet()) {
                System.out.println("  Attribute " + attribute + ":");
                for (Object value : likelihoods.get(label).get(attribute).keySet()) {
                    System.out.println("    Value " + value + ": " + likelihoods.get(label).get(attribute).get(value));
                }
            }
        }
    }

    // Print the counts for each class and class-conditional attribute values
    public void printCounts() {
        // Print class counts
        System.out.println("Class Counts:");
        for (Object label : classCounts.keySet()) {
            System.out.println("Class " + label + ": " + classCounts.get(label) + " instances");
        }

        // Print class-conditional attribute counts
        System.out.println("\nClass-Conditional Attribute Counts:");
        for (Object label : featureCounts.keySet()) {
            System.out.println("Class " + label + ":");
            for (int attribute : featureCounts.get(label).keySet()) {
                System.out.println("  Attribute " + attribute + ":");
                for (Object value : featureCounts.get(label).get(attribute).keySet()) {
                    int count = featureCounts.get(label).get(attribute).get(value);
                    System.out.println("    Value " + value + ": " + count + " instances");
                }
            }
        }
    }

}




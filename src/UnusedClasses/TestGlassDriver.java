import java.util.*;
import java.io.*;

public class TestGlassDriver {

    public static void printTestData(Object[][] testData) {
        System.out.println("Test Data:");
        for (Object[] row : testData) {
            System.out.print("[ ");
            for (Object value : row) {
                System.out.print(value + " ");
            }
            System.out.println("]");
        }
    }
    // Split the dataset into 10 chunks
    public static List<Chunk> splitIntoChunks(Double[][] data, Integer[] labels, int numChunks) {
        List<Object[]> dataset = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            Object[] combined = new Object[data[i].length + 1];
            System.arraycopy(data[i], 0, combined, 0, data[i].length);
            combined[combined.length - 1] = labels[i];
            dataset.add(combined);
        }

        // Shuffle the dataset to ensure randomness
        Collections.shuffle(dataset);

        // Split into chunks
        int chunkSize = dataset.size() / numChunks;
        List<Chunk> chunks = new ArrayList<>();

        for (int i = 0; i < numChunks; i++) {
            Double[][] chunkData = new Double[chunkSize][];
            Integer[] chunkLabels = new Integer[chunkSize];

            for (int j = 0; j < chunkSize; j++) {
                Object[] row = dataset.get(i * chunkSize + j);
                chunkData[j] = toDoubleArray(Arrays.copyOf(row, row.length - 1));
                chunkLabels[j] = (Integer) row[row.length - 1];
            }
            chunks.add(new Chunk(chunkData, chunkLabels));
        }

        return chunks;
    }

    // Helper method to convert Object[] to Double[]
    private static Double[] toDoubleArray(Object[] array) {
        Double[] doubleArray = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            doubleArray[i] = (Double) array[i];
        }
        return doubleArray;
    }

    // Chunk class to hold both data and labels
    static class Chunk {
        Double[][] data;
        Integer[] labels;

        Chunk(Double[][] data, Integer[] labels) {
            this.data = data;
            this.labels = labels;
        }
    }

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/glass.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            // First, count the number of lines to determine the size of the arrays
            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }

            // Reset the reader to the beginning of the file
            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            // Initialize the arrays with the known size
            Integer[] labels = new Integer[lineCount];
            Double[][] data = new Double[lineCount][9]; // Assuming 9 attributes (from column 2 to 10)

            String line;
            int lineNum = 0;

            // Read the file and fill the arrays
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");

                // Assign the label (last column)
                labels[lineNum] = Integer.parseInt(rawData[10]);

                // Fill the data array (columns 1 to 9)
                for (int i = 1; i <= 9; i++) {
                    data[lineNum][i - 1] = Double.parseDouble(rawData[i]);
                }

                lineNum++;
            }

            stdin.close();

            // Split into 10 chunks
            List<Chunk> chunks = splitIntoChunks(data, labels, 10);

            // Loss instance variables
            double totalAccuracy = 0;
            double totalPrecision = 0;
            double totalRecall = 0;
            double totalF1 = 0;
            double total01loss = 0;

            // Perform 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Create training and testing sets
                List<Double[]> trainingData = new ArrayList<>();
                List<Integer> trainingLabels = new ArrayList<>();

                // Extract test chunk
                Chunk testChunk = chunks.get(i);
                Double[][] testData = testChunk.data;
                Integer[] testLabels = testChunk.labels;

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        Chunk trainingChunk = chunks.get(j);
                        trainingLabels.addAll(Arrays.asList(trainingChunk.labels));
                        trainingData.addAll(Arrays.asList(trainingChunk.data));
                    }
                }

                // Convert training data to array form
                Double[][] trainingArray = new Double[trainingData.size()][];
                trainingData.toArray(trainingArray);
                Integer[] trainingLabelsArray = trainingLabels.toArray(new Integer[0]);

                // Train the classifier
                GaussianNaiveBayesClassifier classifier = new GaussianNaiveBayesClassifier(9);  // Assuming 9 attributes
                classifier.train(trainingArray, trainingLabelsArray);

                // Test the classifier
                int correctPredictions = 0;
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;
                for (int j = 0; j < testData.length; j++) {
                    Double[] testInstance = new Double[testData[j].length - 1];
                    System.arraycopy(testData[j], 0, testInstance, 0, testData[j].length - 1);

                    Object predicted = classifier.classify(testInstance);
                    Object actual = testLabels[j];

                        // Print the test data, predicted label, and actual label
                        System.out.print("Test Data: [ ");
                        for (Object feature : testInstance) {
                            System.out.print(feature + " ");
                        }
                        System.out.println("] Predicted: " + predicted + " Actual: " + actual);


                    if (predicted.equals(testLabels[j])) {
                        correctPredictions++;
                    }
                    // Check if the predicted class is 4 (positive class)
                    if (predicted.equals(1)) {
                        if (actual.equals(1)) {
                            truePositives++;  // Correctly predicted class 4 (True Positive)
                        } else {
                            falsePositives++;  // Incorrectly predicted class 4 (False Positive)
                        }
                    } else if (actual.equals(1)) {
                        falseNegatives++;  // Incorrectly predicted something else, but actual is class 4 (False Negative)
                    }
                }
                // Calculate precision and recall for class 4
                double precision = truePositives / (double) (truePositives + falsePositives);
                double recall = truePositives / (double) (truePositives + falseNegatives);
                totalPrecision += precision;
                totalRecall += recall;
                double f1Score;
                if (precision == 0 && recall == 0) {
                    f1Score = 0;  // Avoid NaN
                } else {
                    f1Score = 2 * (precision * recall) / (precision + recall);
                }
                totalF1 += f1Score;
                // Calculate accuracy for this fold
                double accuracy = (double) correctPredictions / testData.length;
                totalAccuracy += accuracy;
                // Calculate 0/1 loss
                double loss01 = 1.0 - (double) correctPredictions / testData.length;
                total01loss += loss01;
                    // Print loss info
                    System.out.println("Number of correct predictions: " + correctPredictions);
                    System.out.println("Number of test instances: " + testData.length);
                    System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
                    System.out.println("Fold " + (i + 1) + " 0/1 loss: " + loss01);
                    System.out.println("Precision for class 1 (fold " + (i + 1) + "): " + precision);
                    System.out.println("Recall for class 1 (fold " + (i + 1) + "): " + recall);
                    System.out.println("F1 Score for class 1 (fold " + (i + 1) + "): " + f1Score);

            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            double average01loss = total01loss / 10;
            double averagePrecision = totalPrecision / 10;
            double averageRecall = totalRecall / 10;
            double averageF1 = totalF1 / 10;
            System.out.println("Average Accuracy: " + averageAccuracy);
            System.out.println("Average 0/1 Loss: " + average01loss);
            System.out.println("Average Precision: " + averagePrecision);
            System.out.println("Average Recall: " + averageRecall);
            System.out.println("Average F1: " + averageF1);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}






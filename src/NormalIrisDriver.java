import java.util.*;
import java.io.*;

//binning
//minor data imputation (we deleted empty line at the end of data set)
//chunks for 10-fold cross validation ARE shuffled in this class
public class NormalIrisDriver {

    // Split the dataset into 10 chunks
    public static List<Object[][]> splitIntoChunks(Object[][] data, Object[] labels, int numChunks) {
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
        List<Object[][]> chunks = new ArrayList<>();

        for (int i = 0; i < numChunks; i++) {
            Object[][] chunk = new Object[chunkSize][];
            for (int j = 0; j < chunkSize; j++) {
                chunk[j] = dataset.get(i * chunkSize + j);
            }
            chunks.add(chunk);
        }

        return chunks;
    }

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/iris.data";
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
            // Get rid of blank line at the bottom of the data set
            lineCount--;
            // Initialize the arrays with the known size
            Object[] labels = new Object[lineCount];
            Object[][] data = new Object[lineCount][4]; // Assuming 4 attributes (from column 1 to 4)

            String line;
            int lineNum = 0;

            // Read the file and fill the arrays
            while ((line = stdin.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;  // Skip this iteration if the line is empty
                }
                String[] rawData = line.split(",");
                // Assign the label (last column)
                labels[lineNum] = rawData[4];


                for (int i = 0; i < rawData.length - 1; i++) {
                    if (Double.parseDouble(rawData[i]) < 1) {
                        data[lineNum][i] = 1;
                    }
                    else if (Double.parseDouble(rawData[i]) < 2) {
                        data[lineNum][i] = 2;
                    }
                    else if (Double.parseDouble(rawData[i]) < 3) {
                        data[lineNum][i] = 3;
                    }
                    else if (Double.parseDouble(rawData[i]) < 4) {
                        data[lineNum][i] = 4;
                    }
                    else if (Double.parseDouble(rawData[i]) < 5) {
                        data[lineNum][i] = 5;
                    }
                    else if (Double.parseDouble(rawData[i]) < 6) {
                        data[lineNum][i] = 6;
                    }
                    else if (Double.parseDouble(rawData[i]) < 7) {
                        data[lineNum][i] = 7;
                    }
                    else if (Double.parseDouble(rawData[i]) < 8) {
                        data[lineNum][i] = 8;
                    }
                }
                lineNum++;
            }

            // print the data to verify
            for (int i = 0; i < lineCount; i++) {
                System.out.print("Label: " + labels[i] + " Data: ");
                for (int j = 0; j < 4; j++) {
                    System.out.print(data[i][j] + " ");
                }
                System.out.println();
            }

            stdin.close();

            // Split into 10 chunks
            List<Object[][]> chunks = splitIntoChunks(data, labels, 10);

            // Loss instance variables
            double totalAccuracy = 0;
            double totalPrecision = 0;
            double totalRecall = 0;
            double totalF1 = 0;
            double total01loss = 0;

            // Perform 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Create training and testing sets
                List<Object[]> trainingData = new ArrayList<>();
                List<Object> trainingLabels = new ArrayList<>();

                Object[][] testData = chunks.get(i);
                Object[] testLabels = new Object[testData.length];
                for (int j = 0; j < testData.length; j++) {
                    testLabels[j] = testData[j][testData[j].length - 1]; // Last column is label
                }

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (Object[] row : chunks.get(j)) {
                            trainingLabels.add(row[row.length - 1]);  // Last column is label
                            Object[] features = new Object[row.length - 1];
                            System.arraycopy(row, 0, features, 0, row.length - 1);
                            trainingData.add(features);
                        }
                    }
                }

                // Convert training data to array form
                Object[][] trainingArray = new Object[trainingData.size()][];
                trainingData.toArray(trainingArray);
                Object[] trainingLabelsArray = trainingLabels.toArray(new Object[0]);

                // Train the classifier
                NaiveBayesClassifier classifier = new NaiveBayesClassifier(4);
                classifier.train(trainingArray, trainingLabelsArray);

                // Test the classifier
                int correctPredictions = 0;
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;
                for (int j = 0; j < testData.length; j++) {
                    Object[] testInstance = new Object[testData[j].length - 1];
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

                    // Get true positives, false positives, and false negatives
                    if (predicted.equals("Iris-virginica")) {
                        if (actual.equals("Iris-virginica")) {
                            truePositives++;
                        } else {
                            falsePositives++;
                        }
                    } else if (actual.equals("Iris-virginica")) {
                        falseNegatives++;
                    }
                }
                // Calculate precision and recall
                double precision = truePositives / (double) (truePositives + falsePositives);
                double recall = truePositives / (double) (truePositives + falseNegatives);
                totalPrecision += precision;
                totalRecall += recall;

                double f1Score = 2 * (precision * recall) / (precision + recall);
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
                System.out.println("Precision for class Iris-virginica (fold " + (i + 1) + "): " + precision);
                System.out.println("Recall for class Iris-virginica (fold " + (i + 1) + "): " + recall);
                System.out.println("F1 Score for class Iris-virginica (fold " + (i + 1) + "): " + f1Score);

            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            double average01loss = total01loss / 10;
            double averagePrecision = totalPrecision / 10;
            double averageRecall = totalRecall / 10;
            double averageF1 = totalF1 / 10;
            System.out.println("Average Accuracy: " + averageAccuracy);
            System.out.println("Average 0/1 Loss: " + average01loss);
            System.out.println("Average Precision for class Iris-virginica: " + averagePrecision);
            System.out.println("Average Recall for class Iris-virginica: " + averageRecall);
            System.out.println("Average F1 for class Iris-virginica: " + averageF1);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}








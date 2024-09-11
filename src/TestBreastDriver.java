import java.util.*;
import java.io.*;

public class TestBreastDriver {

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
        // Assume data and labels are loaded as in your previous driver code
        String inputFile1 = "src/breast-cancer-wisconsin.data";
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
            Object[] labels = new Object[lineCount];
            Object[][] data = new Object[lineCount][9]; // Assuming 9 attributes (from column 2 to 10)

            String line;
            int lineNum = 0;

            // Read the file and fill the arrays
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");

                // Assign the label (first column)
                labels[lineNum] = Integer.parseInt(rawData[10]);

                // Fill the data array (columns 2 to 10)
                for (int i = 1; i <= 9; i++) {
                    if (rawData[i].equals("?")) {
                        data[lineNum][i - 1] = (int) (Math.random() * 10) + 1; // Handle missing values
                    } else {
                        data[lineNum][i - 1] = Integer.parseInt(rawData[i]);
                    }
                }

                lineNum++;
            }

            // print the data to verify
            for (int i = 0; i < lineCount; i++) {
                System.out.print("Label: " + labels[i] + " Data: ");
                for (int j = 0; j < 9; j++) {
                    System.out.print(data[i][j] + " ");
                }
                System.out.println();
            }

            stdin.close();

            // Split into 10 chunks
            List<Object[][]> chunks = splitIntoChunks(data, labels, 10);

            double totalAccuracy = 0;

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
                NaiveBayesClassifier classifier = new NaiveBayesClassifier(9);  // Assuming 9 attributes
                classifier.train(trainingArray, trainingLabelsArray);

                // Test the classifier
                int correctPredictions = 0;
                for (int j = 0; j < testData.length; j++) {
                    Object[] testInstance = new Object[testData[j].length - 1];
                    System.arraycopy(testData[j], 0, testInstance, 0, testData[j].length - 1);

                    Object predicted = classifier.classify(testInstance);
                    if (predicted.equals(testLabels[j])) {
                        correctPredictions++;
                    }
                }

                // Calculate accuracy for this fold
                double accuracy = (double) correctPredictions / testData.length;
                totalAccuracy += accuracy;
                System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
                System.out.println("Number of correct predictions: " + correctPredictions);
                System.out.println("Number of test instances: " + testData.length);
                // Print test data
                printTestData(testData);
            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            System.out.println("Average Accuracy: " + averageAccuracy);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


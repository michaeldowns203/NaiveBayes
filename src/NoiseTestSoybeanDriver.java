import java.util.*;
import java.io.*;

public class NoiseTestSoybeanDriver {
    // Function to shuffle values within a feature column
    public static void shuffleFeature(Object[][] data, int featureIndex) {
        List<Object> featureValues = new ArrayList<>();

        // Extract all values from the feature column
        for (Object[] row : data) {
            featureValues.add(row[featureIndex]);
        }

        // Shuffle the extracted feature values
        Collections.shuffle(featureValues);

        // Put the shuffled values back into the dataset
        for (int i = 0; i < data.length; i++) {
            data[i][featureIndex] = featureValues.get(i);
        }
    }

    // Introduce noise into 10% of the features by shuffling them
    public static void introduceNoise(Object[][] data, int numFeatures) {
        Random rand = new Random();
        Set<Integer> selectedFeatures = new HashSet<>();
        int numNoisyFeatures = (int) Math.ceil(numFeatures * 0.1);  // 10% of features

        // Randomly select 10% of the features
        while (selectedFeatures.size() < numNoisyFeatures) {
            int featureIndex = rand.nextInt(numFeatures);
            selectedFeatures.add(featureIndex);
        }

        // Shuffle values within the selected features
        for (int featureIndex : selectedFeatures) {
            shuffleFeature(data, featureIndex);
        }
    }
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
        String inputFile1 = "src/soybean-small.data";
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
            Object[][] data = new Object[lineCount][35];

            String line;
            int lineNum = 0;

            // Read the file and fill the arrays
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");

                // Assign the label (last column)
                labels[lineNum] = rawData[35];

                for (int i = 0; i < rawData.length - 1; i++) {
                    data[lineNum][i] = Integer.parseInt(rawData[i]);
                }

                lineNum++;
            }
            introduceNoise(data, 35);
            // print the data to verify
            for (int i = 0; i < lineCount; i++) {
                System.out.print("Label: " + labels[i] + " Data: ");
                for (int j = 0; j < 35; j++) {
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
                NaiveBayesClassifier classifier = new NaiveBayesClassifier(35);  // Assuming 9 attributes
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




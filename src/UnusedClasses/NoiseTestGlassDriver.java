import java.util.*;
import java.io.*;

public class NoiseTestGlassDriver {
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
            introduceNoise(data, 9);
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
            List<Chunk> chunks = splitIntoChunks(data, labels, 10);

            double totalAccuracy = 0;

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
                for (int j = 0; j < testData.length; j++) {
                    Double[] testInstance = Arrays.copyOf(testData[j], testData[j].length);
                    Integer trueLabel = testLabels[j];

                    Object predictedLabel = classifier.classify(testInstance);
                    if (predictedLabel.equals(trueLabel)) {
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









import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class NaiveBayesDriverIris {
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
            lineCount--;
            // Initialize the arrays with the known size
            Object[] labels = new Object[lineCount];
            Double[][] data = new Double[lineCount][4]; // Assuming 4 attributes (from column 1 to 4)

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

                // Fill the data array (columns 2 to 10)
                for (int i = 0; i < rawData.length - 1; i++) {
                        data[lineNum][i] = Double.parseDouble(rawData[i]); //arrayindexoutofbounds error here - need to fix
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

            GaussianNaiveBayesClassifier classifier = new GaussianNaiveBayesClassifier(4);
            classifier.train(data, labels);

            Double[] testInstance = {5.0, 4.0, 3.0, 2.0}; //change as well
            Object prediction = classifier.classify(testInstance);
            System.out.println("Predicted class: " + prediction);

        } catch (IOException e) {
            e.printStackTrace();
        }



    }

}
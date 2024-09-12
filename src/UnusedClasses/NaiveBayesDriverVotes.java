

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class NaiveBayesDriverVotes {
    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/house-votes-84.data";
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
            Object[][] data = new Object[lineCount][16]; // Assuming 4 attributes (from column 1 to 4)

            String line;
            int lineNum = 0;

            // Read the file and fill the arrays
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");

                // Assign the label (first column)
                labels[lineNum] = rawData[0];

                // Fill the data array (columns 2 to 10)
                for (int i = 1; i < rawData.length; i++) {
                        data[lineNum][i-1] = rawData[i]; //arrayindexoutofbounds error here - need to fix
                }

                lineNum++;
            }

            // print the data to verify
            for (int i = 0; i < lineCount; i++) {
                System.out.print("Label: " + labels[i] + " Data: ");
                for (int j = 0; j < 16; j++) {
                    System.out.print(data[i][j] + " ");
                }
                System.out.println();
            }

            stdin.close();

            NaiveBayesClassifier classifier = new NaiveBayesClassifier(16);
            classifier.train(data, labels);

            Object[] testInstance = {"y","y","y","y","y","y","y","y","y","y","y","y","y","y","y","y"}; //change as well
            Object prediction = classifier.classify(testInstance);
            System.out.println("Predicted class: " + prediction);

        } catch (IOException e) {
            e.printStackTrace();
        }



    }

}


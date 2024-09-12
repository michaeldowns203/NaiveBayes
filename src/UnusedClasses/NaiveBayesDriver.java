import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class NaiveBayesDriver {
    public static void main(String[] args) throws IOException {
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
                        data[lineNum][i-1] = (int) (Math.random() * 10) + 1; // Handle missing values
                    } else {
                        data[lineNum][i-1] = Integer.parseInt(rawData[i]);
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

            NaiveBayesClassifier classifier = new NaiveBayesClassifier(9);
            classifier.train(data, labels);

            Object[] testInstance = {10, 10, 10, 10, 10, 10, 10, 10, 10};
            Object prediction = classifier.classify(testInstance);
            System.out.println("Predicted class: " + prediction);

        } catch (IOException e) {
            e.printStackTrace();
        }



    }

}

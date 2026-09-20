# Naive Bayes Under Data Noise

An implementation of a discrete-feature **Naive Bayes classifier**, with experiments comparing original and noise-modified datasets. The project explores how data size, class distribution, preprocessing, and noise affect classification performance.

[Read the full paper](FinalPaper.pdf) · [Source code](src/main/)

## Technical highlights

- Class-prior and feature-likelihood estimation with smoothing.
- Dataset-specific parsing, missing-value handling, and binning of continuous features.
- Separate normal and noise-test experiment drivers.
- Ten-fold cross-validation and reporting of 0/1 loss and class-specific F1 scores.

The five datasets are **Wisconsin breast cancer, congressional voting records, soybean, iris, and glass**.

## Run a demonstration

Use a Java JDK with `javac` and `java` available. From the repository root:

```bash
  mkdir -p out
  javac -d out $(find src/main -name '*.java')
  cp -R src/resources/* out/
  java -cp out main.drivers.TestBreastDriver
```

The driver reads `breast-cancer-wisconsin.data`, trains the classifier, and prints model/prediction information for its configured demonstration. Other entry points cover the remaining datasets and noise conditions.

**Reproduction detail:** the inspected `TestBreastDriver` explicitly leaves dataset shuffling commented out, while its comments say shuffling was used for the experimental data. Restore the study's splitting procedure and repetition count before expecting results comparable to the paper. A single demonstration run does not reproduce the reported averages.

## Repository guide

| Location | Purpose |
| --- | --- |
| [`NaiveBayesClassifier.java`](src/main/classifier/NaiveBayesClassifier.java) | Classifier training, prediction, and model inspection. |
| [`TestBreastDriver.java`](src/main/drivers/TestBreastDriver.java) | Breast-cancer demonstration. |
| [`NoiseTestBreastDriver.java`](src/main/drivers/NoiseTestBreastDriver.java) | Breast-cancer noise experiment. |
| [`drivers`](src/main/drivers) | Additional dataset drivers and preprocessing workflows. |

## Limitations

Continuous features are discretized, so results depend on the binning choices. Missing-value imputation and noise generation introduce further variability. The paper identifies limited data and difficult feature distributions as constraints, and notes that some folds did not support the desired F1 calculation.

## Contributors

- **Michael Downs:** Naive Bayes implementation, preprocessing, noise implementation, and loss functions; abstract, introduction, preprocessing, and experimental process.
- **Max Hymer:** ten-fold cross-validation, binning, and classifier debugging; loss functions, algorithm description, results, and conclusions.
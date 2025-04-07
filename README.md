# Gender Classifier using Naive Logistic Regression

## Overview
This project is a gender classification model based on naive logistic regression. It predicts a person's gender based on their name and surname. The model was originally trained on names and surnames commonly found among different nationalities living in Kazakhstan (the dataset was created from data published by Bureau of National Statistics od Kazakhstan). The model can be retrained to accommodate names from other nationalities or regions, however some features (which were specially designed for original dataset) may not be as effective.

## Features
- Predicts gender based on name and surname
- Uses logistic regression for classification
- Extracts n-grams from the beginning and end of names and surnames as features
- Considers name length and number of vowels
- Can be retrained with different datasets to adapt to various nationalities

## Achieved results
- On custom datasets and a part of original dataset average achieved accuracy is 99.5% (due to strong patterns commonly appeared in names of nationalities of Kazakhstan) 

## Usage
1. Prepare your csv file in format name, surname
2. Make predictions:
   ```sh
   python3 main.py <dataset_fname> <predictions_fname> [weights] (if not specified weights.csv is loaded)
   ```
3. Or use interactive mode:
   ```sh
   python3 main.py interactive [weights] (if not specified weights.csv is loaded)
   ```

## Retraining with Different Data
To retrain the model for names from different nationalities:
1. Collect a dataset of names and surnames with gender labels.
2. Format it as a CSV file (`name,surname,gender`).
3. Use the training script to learn from the new dataset.

## Limitations
- The model's accuracy depends on the quality and representativeness of the training data (sinve naive LR is used).
- It may not perform well on names that are uncommon or ambiguous in gender.




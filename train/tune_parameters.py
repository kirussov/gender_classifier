import csv
from utils import load_data
import numpy as np
from classifier_dev import Classifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split

MALE = "male"
FEMALE = "female"
DATASET_FILE = "./data/dataset_ORIGINAL.csv" 
N_SPLITS = 5 # Number of folds (e.g., 5 or 10)
RANDOM_STATE = 42 # For reproducibility of splits

all_data = load_data(DATASET_FILE)
print(f"Loaded {len(all_data)} total records.")

genders = [record['gender'] for record in all_data]
initial_train_data, final_test_data = train_test_split(
        all_data, test_size=0.3, random_state=RANDOM_STATE, stratify=genders
    )
print(f"Split data: {len(initial_train_data)} for training/CV, {len(final_test_data)} for final testing.")

param_grid = {
    'learning_rate': [0.1, 0.05, 0.01], 
    'regularization_rate': [0.01, 0.005, 0.002, 0.001], 
    'iterations': [1000], 
    'ngram_max_len': [5, 6] 
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results = {} 

print(f"\nStarting {N_SPLITS}-Fold Cross-Validation for hyperparameter tuning...")

param_combinations = 0
total_lr = len(param_grid['learning_rate'])
total_reg = len(param_grid['regularization_rate'])
total_iter = len(param_grid['iterations'])
total_ngram = len(param_grid['ngram_max_len'])
total_combinations = total_lr * total_reg * total_iter * total_ngram

print(f"Testing {total_combinations} parameter combinations...")

initial_train_list = list(initial_train_data)

current_combo = 0
for lr in param_grid['learning_rate']:
    for reg in param_grid['regularization_rate']:
         for iters in param_grid['iterations']:
              for ng_max in param_grid['ngram_max_len']:
                    current_combo += 1
                    fold_accuracies = []
                    params = {
                        'learning_rate': lr,
                        'regularization_rate': reg,
                        'iterations': iters,
                        'ngram_max_len': ng_max,
                        'ngram_min_len': 2 
                    }
                    print(f"\n[{current_combo}/{total_combinations}] Testing Params: {params}")

                    fold_num = 0
                    for train_index, val_index in kf.split(initial_train_list):
                        fold_num += 1

                        # create new train ans test samples 
                        train_fold_data = [initial_train_list[i] for i in train_index]
                        val_fold_data = [initial_train_list[i] for i in val_index]

                        # Create and train a new classifier 
                        cv_classifier = Classifier(**params) 
                        cv_classifier.train(train_fold_data)

                        # Evaluate on the test sample
                        accuracy = cv_classifier.evaluate(val_fold_data)
                        fold_accuracies.append(accuracy)
                        print(f"  Fold {fold_num}/{N_SPLITS}, Validation Accuracy: {accuracy:.6f}")

                    # Average accuracy across samples for this parameter set
                    avg_accuracy = np.mean(fold_accuracies)
                    results[tuple(sorted(params.items()))] = avg_accuracy 
                    print(f"-> Average CV Accuracy for {params}: {avg_accuracy:.6f}")


best_params_tuple = max(results, key=results.get)
best_avg_accuracy = results[best_params_tuple]
best_params = dict(best_params_tuple) 

print("\n--- Cross-Validation Results ---")
print(f"Best parameters found: {best_params}")
print(f"Best average cross-validation accuracy: {best_avg_accuracy:.6f}")

print("\nTraining final model using best parameters on the full initial training set...")
final_classifier = Classifier(**best_params)
final_classifier.train(initial_train_data) 

final_weights_file = "final_weights.csv"
final_classifier.save_weights(final_weights_file)

print(f"\n--- Evaluating final model on the held-out test set ({len(final_test_data)} samples) ---")
final_accuracy = final_classifier.evaluate(final_test_data)

print("Final Test Set Classification Report:")
final_predictions = []
final_true_labels = [r['gender'] for r in final_test_data]
for record in final_test_data:
     final_predictions.append(final_classifier.predict(record['first_name'], record['last_name']))
print(classification_report(final_true_labels, final_predictions, labels=[MALE, FEMALE], zero_division=0))

print(f"\nFinal Model Accuracy on Test Set: {final_accuracy:.6f}")

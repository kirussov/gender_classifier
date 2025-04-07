import csv
from math import exp
from random import shuffle
from collections import defaultdict
from utils import get_ngrams, load_data

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MALE = "male"
FEMALE = "female"
FIRST_NAME_PREFIX = "F_"
LAST_NAME_PREFIX = "L_"
BIAS = "_bias_"
VOWELS = ['а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я', 'ә', 'ө', 'ұ', 'ү', 'і']
VOWEL_COUNT_FEATURE = "_vowel_count_value_"

class Classifier:
    def __init__(self, learning_rate, iterations, regularization_rate, 
                 ngram_min_len, ngram_max_len):
        """
            This is a gender classifier, that is able to learn on the given dataset 
            and save weights 
        """
        self.weights = defaultdict(float)

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization_rate = regularization_rate

        self.ngram_min_len = ngram_min_len
        self.ngram_max_len = ngram_max_len
    
    def get_features (self, first_name, last_name) -> set:
        """Extracts features from entered data"""
        features = set()
        fname = first_name.lower()
        lname = last_name.lower()

        features.add(BIAS)

        #names and surnames are features themselves
        features.add(FIRST_NAME_PREFIX + fname)
        features.add(LAST_NAME_PREFIX + lname)

        features.update(get_ngrams(fname, self.ngram_min_len, 
                                   self.ngram_max_len, FIRST_NAME_PREFIX))
        features.update(get_ngrams(lname, self.ngram_min_len, 
                                   self.ngram_max_len, LAST_NAME_PREFIX))
        
        # adds features of typical endings and vowel count
        if fname.endswith(('а', 'я')): features.add(FIRST_NAME_PREFIX + "ends_a_ya")

        if fname and fname[-1] not in VOWELS and fname[-1] != 'ь': 
            features.add(FIRST_NAME_PREFIX + "ends_consonant")

        if lname.endswith(('ова', 'ева', 'ина')): features.add(LAST_NAME_PREFIX + "ends_ova_eva_ina")
        elif lname.endswith(('ая', 'ская')): features.add(LAST_NAME_PREFIX + "ends_aya_skaya")
        elif lname.endswith(('кызы', 'қызы')): features.add(LAST_NAME_PREFIX + "ends_qyzy")
        elif lname.endswith(('улы', 'ұлы')): features.add(LAST_NAME_PREFIX + "ends_uly")


        if lname and lname[-1] not in VOWELS and lname[-1] != 'ь': 
            features.add(LAST_NAME_PREFIX + "ends_consonant")

        # extracts name's and surname's length feature
        f_len = len(fname)
        if f_len <= 4: features.add(FIRST_NAME_PREFIX + "len_0_4")
        elif f_len <= 7: features.add(FIRST_NAME_PREFIX + "len_5_7")
        else: features.add(FIRST_NAME_PREFIX + "len_8_plus")

        l_len = len(lname)
        if l_len <= 5: features.add(LAST_NAME_PREFIX + "len_0_5")
        elif l_len <= 9: features.add(LAST_NAME_PREFIX + "len_6_9")
        else: features.add(LAST_NAME_PREFIX + "len_10_plus")

        return features
    
    def get_vowel_data(self, first_name, last_name) -> int:
        """Counts vowels in name and surname"""
        count = 0
        for char in first_name.lower():
            if char in VOWELS: count += 1

        for char in last_name.lower():
            if char in VOWELS: count += 1

        return count 
    
    def get_score(self, features, vowel_count_value) -> float:
        """Calculates score using feature weights"""
        score = 0.0

        for feature in features:
            score += self.weights[feature]

        score += vowel_count_value * self.weights[VOWEL_COUNT_FEATURE]

        return score
    
    def get_probability (self, score) -> float:
        """
            Calculates probability of name + surname combination being female
            using sigmoidal function
        """
        if score < -700: return 0.0
        elif score > 700: return 1.0
        else: return (1 / (1 + exp(-score)))

    def train (self, training_data) -> None:
        """Iterational learning"""
        for _ in range (self.iterations):
            shuffle(training_data)

            for person in training_data:
                first_name = person["first_name"]
                last_name = person["last_name"]
                gender = person["gender"]

                features = self.get_features(first_name, last_name)
                vowel_count_val = self.get_vowel_data(first_name, last_name)
                score = self.get_score(features, vowel_count_val)
                prediction = self.get_probability(score)

                truth = 1 if gender == FEMALE else 0
                error = prediction - truth

                for feature in features:
                    if feature != BIAS:
                        regularization = self.regularization_rate * self.weights[feature]
                    else:
                        regularization = 0

                    gradient = error + regularization
                    self.weights[feature] -= self.learning_rate * gradient
    
    def predict (self, first_name, last_name) -> str:
        """Predicts gender based on the weights"""
        features = self.get_features(first_name, last_name)
        vowel_count_val = self.get_vowel_data(first_name, last_name)
        score = self.get_score(features, vowel_count_val)

        probability = self.get_probability(score)
        prediction = FEMALE if probability >= 0.5 else MALE

        return prediction
    
    def save_predictions (self, filename, data) -> None:
        """Saves predicted genders to csv file"""
        with open (filename, "w", encoding="utf-8") as file:
            writer = csv.writer(file)

            for person in data:
                fname = person["first_name"]
                lname = person["last_name"]
                writer.writerow([fname, lname, self.predict(fname, lname)])
            
    def save_weights (self, filename="weights.csv") -> None:
        """Saves weights to csv file"""
        with open (filename, "w", encoding="utf-8") as file:

            writer = csv.writer(file)
            for feature, weight in self.weights.items():
                writer.writerow([feature, weight])

    def load_weights (self, filename="weights.csv") -> None:
        """Loads weights from csv file"""
        self.weights = defaultdict(float)
        
        with open (filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            for row in reader:
                self.weights[row[0]] = (float)(row[1])

    def evaluate(self, test_data):
        """Evaluates the model on test data and prints results"""
        predictions = []
        true_labels = []

        for record in test_data:
            prediction = self.predict(record['first_name'], record['last_name'])
            predictions.append(prediction)
            true_labels.append(record['gender'])

        print("\n--- Evaluation Report ---")
        print(classification_report(true_labels, predictions, labels=[MALE, FEMALE], zero_division=0))
        print("-------------------------\n")

        # Classic accuracy calculation
        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        accuracy = correct / len(true_labels)
        print(f"Overall Accuracy: {accuracy:.4f}")
        return accuracy



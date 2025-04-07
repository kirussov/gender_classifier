import csv
from math import exp
from collections import defaultdict
from utils import get_ngrams

MALE = "male"
FEMALE = "female"
FIRST_NAME_PREFIX = "F_"
LAST_NAME_PREFIX = "L_"
BIAS = "_bias_"
VOWELS = ['а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я', 'ә', 'ө', 'ұ', 'ү', 'і']
VOWEL_COUNT_FEATURE = "_vowel_count_value_"
SUS_RANGE = 0.1

class Classifier:
    """
        This is a gender classifier based on naive logistic regression. 
        Some features were specially designed to correspond common patterns 
        from names and surnames of nationalities living in Kazakhstan.
    """
    def __init__(self, ngram_min_len, ngram_max_len):
        self.weights = defaultdict(float)

        self.suspicious_predictions = list()
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

        if lname.endswith(('ова', 'ева', 'ина')): 
            features.add(LAST_NAME_PREFIX + "ends_ova_eva_ina")
        elif lname.endswith(('ая', 'ская')): 
            features.add(LAST_NAME_PREFIX + "ends_aya_skaya")
        elif lname.endswith(('кызы', 'қызы')): 
            features.add(LAST_NAME_PREFIX + "ends_qyzy")
        elif lname.endswith(('улы', 'ұлы')): 
            features.add(LAST_NAME_PREFIX + "ends_uly")


        if lname and lname[-1] not in VOWELS and lname[-1] != 'ь': 
            features.add(LAST_NAME_PREFIX + "ends_consonant")

        # extracts name's and surname's length feature
        f_len = len(fname)
        if f_len <= 4: 
            features.add(FIRST_NAME_PREFIX + "len_0_4")
        elif f_len <= 7: 
            features.add(FIRST_NAME_PREFIX + "len_5_7")
        else: 
            features.add(FIRST_NAME_PREFIX + "len_8_plus")

        l_len = len(lname)
        if l_len <= 5: 
            features.add(LAST_NAME_PREFIX + "len_0_5")
        elif l_len <= 9: 
            features.add(LAST_NAME_PREFIX + "len_6_9")
        else: 
            features.add(LAST_NAME_PREFIX + "len_10_plus")

        return features
    
    def get_vowel_data(self, first_name, last_name) -> int:
        """Counts vowels in name and surname"""
        vow_count = 0
        for char in first_name.lower():
            if char in VOWELS: vow_count += 1

        for char in last_name.lower():
            if char in VOWELS: vow_count += 1

        return vow_count 
    
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

    def predict (self, first_name, last_name) -> str:
        """Predicts gender based on the weights"""
        features = self.get_features(first_name, last_name)
        vowel_count_val = self.get_vowel_data(first_name, last_name)
        score = self.get_score(features, vowel_count_val)

        probability = self.get_probability(score)
        prediction = FEMALE if probability >= 0.5 else MALE

        if abs(probability - 0.5) <= SUS_RANGE: 
            self.suspicious_predictions.append([first_name, last_name, prediction, probability])

        return prediction
    
    def save_predictions (self, filename, data) -> None:
        """Saves predicted genders to csv file"""
        with open (filename, "w", encoding="utf-8") as file:
            writer = csv.writer(file)

            for person in data:
                fname = person["first_name"]
                lname = person["last_name"]
                writer.writerow([fname, lname, self.predict(fname, lname)])

        with open ("SUSPICIOUS_" + filename, "w", encoding="utf-8") as file:
            writer = csv.writer(file)

            for person in self.suspicious_predictions:
                writer.writerow([person[0], person[1], person[2], person[3]])
 
    def load_weights (self, filename="weights.csv") -> None:
        """Loads weights from csv file"""
        self.weights = defaultdict(float)
        
        with open (filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)

            for row in reader:
                self.weights[row[0]] = (float)(row[1])
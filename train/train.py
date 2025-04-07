import csv
from sys import argv
from utils import load_data
from classifier_dev import Classifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

if len(argv) < 2: 
    print("Please enter the path to the dataset!")
    exit()
    
path_to_dataset = argv[1]
data = load_data(path_to_dataset)

classifier = Classifier(
            learning_rate=0.1,       
            iterations=1000,          
            regularization_rate=0.01, 
            ngram_min_len=2,
            ngram_max_len=6,        
        )

genders = [record['gender'] for record in data]
train_data, test_data = train_test_split (data, 
                                          test_size=0.3, 
                                          random_state=42, 
                                          stratify=genders)

print(f"Data splitted: {len(train_data)} for training, {len(test_data)} for test")

print("Starting training...")
classifier.train(train_data)

classifier.evaluate(test_data)

if input("Would you like to save weights? (y/n)") == "y":
    classifier.save_weights("weights_evaluated.csv")

if input("Would you like to generate weights from the whole dataset without evaluation? (y/n)") == "y":
    classifier = Classifier(
            learning_rate=0.1,       
            iterations=1000,          
            regularization_rate=0.01, 
            ngram_min_len=2,
            ngram_max_len=6,        
        )
    classifier.train(data)
    classifier.save_weights("weights_whole.csv")

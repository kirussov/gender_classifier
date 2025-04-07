from sys import argv
from classifier import Classifier
from utils import load_data

classifier = Classifier(ngram_min_len=2, ngram_max_len=6)

if len(argv) >= 3 and argv[1] != "interactive":
    src = argv[1]
    dest = argv[2]
    weights_path = argv[3] if len(argv) >= 4 else "weights.csv"
    classifier.load_weights(weights_path)

    data_to_predict = load_data(src)
    classifier.save_predictions(dest, data_to_predict)


elif len(argv) >= 2 and argv[1] == "interactive":
    weights_path = argv[2] if len(argv) >= 3 else "weights.csv"
    classifier.load_weights(weights_path)
    print(f"Interactive mode: Weights loaded from {weights_path}")
    print("Entering interactive mode. Press Ctrl+C to exit.")
    while True:
            fname = input("First name: ")
            lname = input("Last name: ")
            prediction = classifier.predict(fname, lname)
            print(f"-------\nPredicted Gender: {prediction}\n-------")

else:
    print("Usage:")
    print("  For file prediction: python main.py <input_csv> <output_csv> [weights_file.csv]")
    print("  For interactive mode: python main.py interactive [weights_file.csv]")
    print("  (Default weights file is 'weights.csv')")
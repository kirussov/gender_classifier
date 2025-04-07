import csv

def get_ngrams (data, min_len, max_len, prefix="") -> set:
    """Extracts n-grams from entered data in the given length range"""
    result = set()
    data = data.lower()

    for n in range (min_len, max_len + 1):
        if len(data) >= n:
            result.add(prefix + "pre_" + data[:n])
            result.add(prefix + "suf_" + data[-n:])

    return result

def load_data (filename) -> list:
    """Reads data from csv file and returns a list of dictionaries"""
    data = list()

    with open (filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append({"first_name": row[0],
                         "last_name": row[1],
                         "gender": row[2]})
    
    return data

import csv
import random
import math

def load_csv(filename):
    with open(filename, "r") as file:
        return [list(map(float, row)) for row in csv.reader(file)]

def split_dataset(dataset, ratio):
    train_size = int(len(dataset) * ratio)
    train_set = random.sample(dataset, train_size)
    return train_set, [row for row in dataset if row not in train_set]

def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        separated.setdefault(row[-1], []).append(row)
    return separated

def summarize(dataset):
    return [(mean(col), stdev(col)) for col in zip(*dataset[:-1])]

def mean(numbers):
    return sum(numbers) / len(numbers)

def stdev(numbers):
    avg = mean(numbers)
    variance = sum((x - avg) ** 2 for x in numbers) / (len(numbers) - 1)
    return math.sqrt(variance)

def calculate_probability(x, mean, stdev):
    if stdev == 0:
        return 1 if x == mean else 0
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, stats in summaries.items():
        prob = 1
        for i, (mean, stdev) in enumerate(stats):
            prob *= calculate_probability(input_vector[i], mean, stdev)
        probabilities[class_value] = prob
    return probabilities

def predict(summaries, input_vector):
    probs = calculate_class_probabilities(summaries, input_vector)
    return max(probs, key=probs.get)

def get_predictions(summaries, test_set):
    return [predict(summaries, row) for row in test_set]

def get_accuracy(test_set, predictions):
    correct = sum(1 for i in range(len(test_set)) if test_set[i][-1] == predictions[i])
    return (correct / len(test_set)) * 100.0
def main():
    dataset = load_csv('diabetes.csv')
    train_set, test_set = split_dataset(dataset, 0.87)
    print(f'Split {len(dataset)} rows into training={len(train_set)} and testing={len(test_set)} rows')
    
    summaries = {cls: summarize(rows) for cls, rows in separate_by_class(train_set).items()}
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print(f'Classification Accuracy: {accuracy:.2f}%')
main()

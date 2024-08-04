import csv

def find_s_algorithm(filename):
    with open(filename, 'r') as csv_file:
        data = [line for line in csv.reader(csv_file) if line[-1] == "Yes"]

    if not data:
        print("\nNo positive examples found.")
        return

    hypo = data[0][:-1]
    
    for example in data[1:]:
        hypo = [h if h == e else '?' for h, e in zip(hypo, example[:-1])]
    
    print("\nMaximally specific Find-S hypothesis:")
    print(hypo)

# Example usage
find_s_algorithm('enjoysport.csv')

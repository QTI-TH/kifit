import click
import numpy as np

@click.command()
@click.option('--datapath', required=True, help='Path to the data file')
@click.option('--cols', required=True, type=int, nargs=2, help='Two column indices to swap (0-based)')
@click.option('--keyword', default="Xcoeff(Hz):", type=str, help="After keyword there are real data columns")
def process_file(datapath, cols, keyword):

    col1, col2 = cols
    data = np.loadtxt(datapath)

    with open(datapath, 'r') as file:
        first_line = file.readline().strip()

    keyword_found = False
    new_headline = ""
    labels = []

    for word in first_line.split():
        if keyword in word:
            keyword_found = True
        if keyword_found:
            labels.append(word)
        else:
            new_headline += f"{word}\t"
        
    labels[col1], labels[col2] = labels[col2], labels[col1]
    data[:, [col1, col2]] = data[:, [col2, col1]]

    for label in labels:
        new_headline += f"{label}\t"

    outputfile = f"""{datapath.removesuffix(".dat")}_swapped_{col1}_{col2}.dat"""

    with open(outputfile, 'w') as file:
        file.write(new_headline + '\n')
        np.savetxt(file, data, delimiter='\t')

if __name__ == '__main__':
    process_file()
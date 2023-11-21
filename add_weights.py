import os
import pandas as pd


def main():
    folder_path = "./Cisco_22_networks/dir_20_graphs/dir_day1"
    for day_data in os.listdir(folder_path):
        if day_data.endswith('.txt'):
            file_path = os.path.join(folder_path, day_data)

            df = pd.read_csv(file_path, sep='\s+', header=None)

            third_column = df.iloc[:, 2]
            comm_info = third_column.str.split(',', expand=True)

            

    pass


if __name__ == '__main__':
    main()

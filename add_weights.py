import os
import pandas as pd
from typing import Dict, TypedDict


def main():
    folder_path = "./Cisco_22_networks/dir_20_graphs/dir_day1"
    for day_data in os.listdir(folder_path):
        if day_data.endswith('.txt'):
            file_path = os.path.join(folder_path, day_data)
            # create a df delimited by space char
            df = pd.read_csv(file_path, sep='\s+', header=None)
            # get series rep the column
            # might look like:
            #     0    "a,b,c"
            #     1    "d,e"
            #     2    "f"
            # where a,b,c..etc. is some comm over unique port+protocol
            third_column = df.iloc[:, 3]
            # splitting + expanding to a full df is a bad idea
            # cant create jaggeded df so it has a bunch of NaNs
            split_values = third_column.str.split(',')
            list_of_lists = split_values.tolist()

            for i, client_server_comm in enumerate(list_of_lists):
                # use index for when you go back to df
                ports_data: Dict[int, PortData] = {}
                for port_protocol_comm in client_server_comm:
                    # port_protocol_comm example : 1p6-22
                    # where 1 is port number,
                    # 6 is protocol num and 22 is packet size
                    port_protocol_comm = str(port_protocol_comm.strip())
                    # direct accesses would fail. need  to split
                    # ex: if you specify protocol is index 2 but its
                    #     actually protocol 17 youd miss info
                    arr = port_protocol_comm.split("p")
                    assert (len(arr) == 2)
                    port = int(arr[0])
                    arr = arr[1].split("-")
                    assert (len(arr) == 2)
                    protocol = int(arr[0])
                    packet_size = int(arr[1])

                    if port not in ports_data:
                        ports_data[port] = PortData(
                            total_packet_size=packet_size,
                            appearance_count=1)
                    else:
                        ports_data[port]['appearance_count'] += 1
                        ports_data[port]['total_packet_size'] += packet_size

                # now we can go through it all
                final_weight = 0
                for port, port_data in ports_data.items():
                    final_weight += port_data['appearance_count'] * \
                        port_data['total_packet_size']
                df.loc[i, 'weight'] = final_weight
            df.to_csv(os.path.join(
                folder_path, day_data.split('.txt')[0])+".csv")


class PortData(TypedDict):
    total_packet_size: int
    appearance_count: int


if __name__ == '__main__':
    main()

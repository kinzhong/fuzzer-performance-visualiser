import os

directory = r'D:\Users\KinZhong\Dropbox\Computer Science\Final Year Project\fuzzer-performance-visualiser\toml-files'
for filename in os.listdir(directory):
    print("sample-toml\\" + filename)
    os.system("python3 visualiser.py sample-toml\\" + filename)




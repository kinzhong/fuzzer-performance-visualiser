# Fuzzer Performance Visualiser
The fuzzer performance visualiser is a simple tool that can generate highly configurable visualisations from showmap data. The visualiser generates the visualisations in various forms such as a PDF, PNG, and interactive [HTML](https://htmlpreview.github.io/?https://github.com/kinzhong/fuzzer-performance-visualiser/blob/main/sample-visualisations/readelf-with-legend/readelf-mean-edge-time.html).

A sample visualisation can be seen below.
![performance evaluation using readelf](https://github.com/kinzhong/fuzzer-performance-visualiser/blob/main/sample-visualisations/readelf-with-legend/readelf-mean-edge-time.png)

# Dependencies
The dependencies are stored in requirements.txt.

The requirements can be installed by running the following command:
```
pip install -r requirements.txt
```

# Usage
After running the fuzzers, the raw data pertaining to the performance of the fuzzer can be generated using the showmaps tool from [fuzzer-data-collector](https://github.com/ThePatrickStar/fuzzer-data-collector).

Next, the visualiser takes in a TOML configuration file, which contains the file location of the generated raw data and the visualisation configuration options. A example TOML can be found from [sample-toml/example.toml](sample-toml/example.toml). The [sample-toml/](sample-toml/) directory also contains other samples which also be generated and used as examples.

Lastly, visualisations can be generated using visualiser.py.
```
python3 visualiser.py [TOML_FILE]
```

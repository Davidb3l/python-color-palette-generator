# Video frame color palette generator

Modified to extract a specific frame by timestamp.

![Screenshot](https://gauracs.me/wp-content/uploads/2020/12/color_palette_generator_frame.jpg)

##### Example of the result image
![Screenshot](https://gauracs.me/wp-content/uploads/2020/12/color_palette_generator.jpg)

### Prerequisites

In order to run this program, several Python packages must be installed first using pip and Homebrew:

```
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install opencv-python
```

### Run the program
Example command:

```
python .\main.py --vid sample.mp4 --frame 00:01:00  
```

## License

MIT

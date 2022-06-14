# Augmented Reality

"ar.py" is a python program that uses OpenCV to render .obj files on Aruco markers. For an in-depth explanation, check out my [blog tutorial](https://medium.com/swlh/augmented-reality-diy-3fc138274561).

<!-- ![Demo](https://user-images.githubusercontent.com/31953115/121981314-0b712c00-cdab-11eb-98d4-decf737824ea.gif) -->

<p align = "center">
  <img src="https://user-images.githubusercontent.com/31953115/121981314-0b712c00-cdab-11eb-98d4-decf737824ea.gif" alt="animated" />
</p>


## Dependencies

It uses OpenCV(3.4.2) and Numpy(1.17.2). The python version is 3.7.5

Before installing the dependencies, it is recommended to create a virual environment to prevent confilcts with the existing environment. Using conda, 

```bash
conda create -n augmented_reality python=3.7.5
conda activate augmented_reality
``` 

To install the dependencies - 
```bash
pip install -r requirements.txt
```

## Usage
For the main program - 

```bash
python ar.py
```
It is recommended to print the aruco marker(data/m1.pdf) on a piece of paper. Alternatively, the program should work (but inferiorly) with the marker open on your phone. The white margin around the marker boundary is required for boundary detection. Keeping the marker flat (for example, by sticking it to a piece of cardboard) further helps in detecting the marker. 

For a faster version which uses the Lucas-Kanade method for tracking - 
```bash
python ar_with_tracking.py
```

(Optional) For camera calibration alter the path mentioned in the file and run - 

```bash
python camera_calib.py 
```

Refer to the blog for all the details about camera calibration. 


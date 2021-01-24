### Rozpoznawanie emocji tłumów metodami uczenia maszynego
### Analysis of crowd emotions with machine learning methods

The purpose of this engineering thesis was to train a model capable of recognizing the emotions of crowds in photos.


#### Installation
The application requires a Python interpreter. The source code was written for version 3.7, which along with others can be downloaded [here](https://www.python.org/downloads/) for Windows. For Linux, the Python 3.7 interpreter can be installed using the command: ```sudo apt install python3.7```

After installing Python, download the application's source code and go to its main folder. There, using the command below, create a folder named ```venv``` containing the virtual environment:  
Windows: ```python -m venv venv```  
Linux: ```python3 -m venv venv```

Each time you start working with the application, activate the virtual environment:  
Windows: ```venv\Scripts\activate.bat```  
Linux: ```source venv/bin/activate```

At this point in the terminal ```(venv)``` should appear before the path to the current directory. From now on, the commands ```python``` and ```pip``` (for Windows) as well as ```python3``` and ```pip3``` (for Linux) will use the virtual environment. If it was activated for the first time, install all necessary libraries and dependencies required by the applications with the command:  
Windows: ```pip install -r requirements.txt```  
Linux: ```pip3 install -r requirements.txt```

If you have a GPU, instead of ```requirements.txt``` you can also use ```requirements-gpu.txt``` to install the TensorFlow library in a GPU capable version. In this case, it is also necessary to install the Nvidia cuDNN drivers, which you can download [here](https://developer.nvidia.com/cudnn). After executing all the above commands, the application is ready to work.

#### Usage
For each new terminal session, the virtual environment must be activated before both the neural network training application and the emotion recognition script are run for the first time. This can be done with the command:  
Windows: ```venv\Scripts\activate.bat```  
Linux: ```source venv/bin/activate```

The network training application can be launched using:  
Windows: ```python src\train_main.py```  
Linux: ```python3 src/train_main.py```
    
It accepts no arguments. Instead, it is controlled by the ```train_config.json``` configuration file. All parameters of this file, as well as the way the application works, are described in detail in chapter 5.

The emotion recognition script can be run with:  
Windows: ```python src\predict_main.py path1 path2 path3...```  
Linux: ```python3 src/predict_main.py path1 path2 path3...```

It takes any positive number of arguments that stores the paths to files and folders containing .jpg images to be analyzed. The remaining script parameters can be configured using the file ```predict_config.json```. They are described in detail in Chapter 5, along with the way the script works and what it can do.

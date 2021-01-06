# Watching TV Action Detection in videos using Deep LearningYours sincerely
Deep Learning model that detects the action of Watching TV in a video

# Install Dependencies

    pandas
    numpy
    scipy
    sklearn
    tensorflow-gpu
    keras
    matplotlib

# Execution

# Generate landmarks:

    Videos are submitted to the OpenPose portable execution file to generate JSON files of the body landmarks, and the left and right hand landmarks for each frame in the video.

# Preprocess data:

    Read the JSON files (landmarks for each frame in the video) in every folder.
    Collect the full body (pose), and the left and right hand landmarks.
    Calculate the euclidean distances between body pose landmarks for each frame
    Calculate the euclidean distances between left hand landmarks for each frame
    Calculate the euclidean distances between right hand landmarks for each frame       
    Output the file for each video.

    The above operations can be executed by running 'parse_json_tv.ipynb'

# Deep Learning model:

    Read all training data files and append them.
    Read the test data file.
    Normalize the features for both training and test data in the interval [0,1].
    Fit the Deep Learning model on training data.
    Test the model on testing data.
    Save Time vs Label JSON file and graph.

    The above operations can be executed by running 'tv_dnn_model.ipynb'

# Trained model:

    The trained models are available in the below path

    DNN_watching_TV_action_detection/Trained_model

# Code Demo Video Link:

    https://youtu.be/hDs4ejh7kMY  

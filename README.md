# DocumentClassification

## Data preparation 

### Data collection

I used the chromium engine to automate the download of the data using the google image `API`. 
I applied ImageDataGenerator, a class of the keras `API`, to download each image and modify the distribution of its pixels (zoom, rotation...) for two reasons: on the one hand, to reduce the size of the data and, on the other hand, so that our model is more efficient on general data.    

### Data processing
I have 3792 examples of each category, they must be equal in quantity to keep the same probabilities.

Each example from the data contains image resized to `128x128` with colors black and white using binairzation technique, its values are divided by 255 so each image contains values from 0 to 1 with type float. 

The training example also contains a label which can be (facture, devis, bon (bon de commande), cheque and lettre), I converted the string label to number like `{"0": "bon", "1": "devis", "2": "facture", "3": "lettre", "4": "cheque" }`. I also applied on hot encoding on each label so that each label becomes an array of binary.
example : `Label => "facture" =>3 => [0,1,1,0,1] `

I divided my data by 0.75 for the training set from whom our model will learn patterns and 0.25 for the test set where our model will evaluate its learning process. 
The data should be shuffled, i.e. the position of each training example should be random. 

### Training
The model is built with an input layer (convolution layer), 11 hidden layers and an output layer (dense layer). 
  - The input must be of size `128x128` 
  - The hyper-parameters defined by : the activation function for the input and the hidden layer will be Relu, for the output layer softmax is used.
  - I used 6 epochs and 64 as a batch size for training, which means that in each epoch will train the total number of training examples divided by 64. 
  - Loss function is used : categorical_crossentropy because we have 5 categories to predict 
  - The optimizer will be "adam".
  - The regularization will be dropout  

## Usage
The following instructions should be implemented in this order :
- Clone the repository by executing `git clone https://github.com/Youssef1998d/DocumentClassification.git`.
- Download the zip file in this link `https://drive.google.com/file/d/1taTW475vHMYLN6_HlypOOsahhByKUABP/view?usp=sharing`, extract it in cloned folder.
- To generate the `cnn_classification_model`, run `python3 create_cnn_model.py` in shell or `python create_cnn_model.py` in Windows/MacOS.
- In the shell (or Windows/MacOS) the following command should be executed `pip install -r requirements.txt` (of course within the folder path).

### Predict a document type
The script execution in shell `python3 predit.py` (or in Windows `python3 predit.py`) then the user should input the absolute path of the image, then the output will be printed as the class of the document.


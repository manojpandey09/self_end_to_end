#coming from training_pipeline.py file , now we will code here 

#by running this code artifacts folder will be created with 3 csv files

from src.DiamondPricePrediction.components.data_ingestion import DataIngestion   # Handles dataset loading & splitting
from src.DiamondPricePrediction.components.data_transformation import DataTransformation  # Handles preprocessing
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer    # Handles ML model training
from src.DiamondPricePrediction.exception import customexception  # Custom exception handling
import sys
import pandas as pd 

obj= DataIngestion()
obj.initiate_data_ingestion()


#now copy the model trainer.py file path and run u can see the artifacts folder with 3 csv files stored in it


#now we moving to data_ransformation.py file




#we cacame here from utils file now we will add lines here 

train_data_path, test_data_path = obj.initiate_data_ingestion()
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_trainer(train_arr, test_arr)


#run the code python traiing pipeline.py with path
# u will see a model.pkl file and 

#after running code u will see a model.pkl file and preprocessor file is  created in artifacts folder
#and u wll see a result with model training is completed
#now move to prediction.pipeline file and code







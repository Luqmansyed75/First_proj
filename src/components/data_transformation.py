import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
            this func is responsible for data transformation
        '''
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical cols standard scaling completed")
            logging.info("Categorical cols encoding completed")

            preprocessor=ColumnTransformer(
                transformers=[
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("Categorical_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("obtaining preprocesisng objects")

            preprocesisng_obj=self.get_data_transformer_obj()

            target_column_name="math_score"
            numerical_columns=["reading_score","writing_score"]

            input_features_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"applying preprocessing obj on training dataframe and testing dataframe"
            )

            input_features_train_arr=preprocesisng_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocesisng_obj.transform(input_features_test_df)

            train_arr=np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("savaed processing obj")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocesisng_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    from src.logger import logging

    logging.info("Starting data transformation testing...")

    obj = DataTransformation()
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    train_arr, test_arr, _ = obj.intiate_data_transformation(train_path, test_path)

    logging.info("Data transformation completed successfully!")
    print("Transformation complete. Check logs for details.")


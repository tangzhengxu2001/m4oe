import os
import argparse
import numpy as np
import pandas as pd
import cv2
import pydicom

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import Counter
from tqdm import tqdm
from PIL import Image

def set_cancer_risk_label(df, risk_years):
    """
    Add a cancer risk label to the dataframe based on the years to cancer diagnosis.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing a 'year_to_cancer' column.
        risk_years (int): Threshold year value. Patients with years to cancer diagnosis
                          less than or equal to risk_years are labeled as 1, otherwise 0.
    
    Returns:
        pd.DataFrame: DataFrame with a new column 'cancer_risk_label{risk_years}'.
    """
    # Apply a lambda function to assign risk label: 1 if diagnosis is within risk_years, else 0
    df[f'cancer_risk_label{risk_years}'] = df['year_to_cancer'].apply(lambda x: 1 if x <= risk_years else 0)
    return df

def process_embed_data(metadata_file,
                       clinical_file,
                       prefix_path,
                       fixed_prefix="/mnt/NAS2/mammo/anon_dicom/",
                       primary_key=['empi_anon', 'study_date_anon'],
                       image_cols=['anon_dicom_path', 'FinalImageType', 'ViewPosition'],
                       disease_cols=['tissueden', 'asses', 'path_severity']):
    """
    Process and merge metadata and clinical CSV files into a single dataframe for embedding.
    
    This function reads metadata and clinical data from CSV files, filters relevant columns,
    adjusts file paths for images, groups image data based on image type and view position,
    and then merges the processed metadata with the clinical data.
    
    Parameters:
        metadata_file (str): Path to the metadata CSV file.
        clinical_file (str): Path to the clinical CSV file.
        prefix_path (str): Path prefix to be added to the image file paths.
        fixed_prefix (str): Prefix string in the original file paths to be removed.
        primary_key (list): List of columns used as primary keys for merging.
        image_cols (list): List of columns in the metadata that pertain to image information.
        disease_cols (list): List of columns in the clinical file related to disease information.
    
    Returns:
        pd.DataFrame: Merged dataframe containing both image and clinical data.
    """
    # Read metadata and clinical CSV files
    metadata = pd.read_csv(metadata_file)
    clinical = pd.read_csv(clinical_file)

    # Select only the necessary columns for metadata and clinical data
    metadata = metadata[primary_key + image_cols]
    clinical = clinical[primary_key + disease_cols]

    # Adjust image file paths by removing the fixed prefix and adding the prefix_path
    metadata['anon_dicom_path'] = metadata['anon_dicom_path'].apply(
        lambda x: os.path.join(prefix_path, x.replace(fixed_prefix, "")) if isinstance(x, str) else x
    )

    # Process images for "cview" type with CC view
    df_c_view_cc = (metadata[(metadata['FinalImageType'] == 'cview') & (metadata['ViewPosition'] == 'CC')]
                    [primary_key + ['anon_dicom_path']]
                    .rename(columns={'anon_dicom_path': 'c_view_cc'}))
    # Keep the first occurrence for each primary key combination
    df_c_view_cc = df_c_view_cc.groupby(primary_key, as_index=False).first()
    
    # Process images for "cview" type with MLO view
    df_c_view_mlo = (metadata[(metadata['FinalImageType'] == 'cview') & (metadata['ViewPosition'] == 'MLO')]
                     [primary_key + ['anon_dicom_path']]
                     .rename(columns={'anon_dicom_path': 'c_view_mlo'}))
    df_c_view_mlo = df_c_view_mlo.groupby(primary_key, as_index=False).first()
    
    # Process images for "2D" type with CC view
    df_2d_cc = (metadata[(metadata['FinalImageType'] == '2D') & (metadata['ViewPosition'] == 'CC')]
                [primary_key + ['anon_dicom_path']]
                .rename(columns={'anon_dicom_path': '2d_cc'}))
    df_2d_cc = df_2d_cc.groupby(primary_key, as_index=False).first()
    
    # Process images for "2D" type with MLO view
    df_2d_mlo = (metadata[(metadata['FinalImageType'] == '2D') & (metadata['ViewPosition'] == 'MLO')]
                 [primary_key + ['anon_dicom_path']]
                 .rename(columns={'anon_dicom_path': '2d_mlo'}))
    df_2d_mlo = df_2d_mlo.groupby(primary_key, as_index=False).first()
    
    # Merge all image dataframes on the primary keys using outer joins
    df_merged = df_c_view_cc.merge(df_c_view_mlo, on=primary_key, how='outer')
    df_merged = df_merged.merge(df_2d_cc, on=primary_key, how='outer')
    df_merged = df_merged.merge(df_2d_mlo, on=primary_key, how='outer')

    # Ensure study_date_anon is in datetime format for both dataframes if it is part of the primary key
    if 'study_date_anon' in primary_key:
        df_merged['study_date_anon'] = pd.to_datetime(df_merged['study_date_anon'], errors='coerce')
        clinical['study_date_anon'] = pd.to_datetime(clinical['study_date_anon'], errors='coerce')
    
    # Merge the image metadata with the clinical data using a left join
    df_final = df_merged.merge(clinical, on=primary_key, how='left')
    
    return df_final


class MammographyDataset_complete(Dataset):
    """
    PyTorch Dataset for loading mammography images along with their corresponding labels.
    
    This dataset reads processed data from a CSV file, filters the data based on valid labels,
    applies transformations to the images, and returns tuples containing multiple image views,
    target labels, and record identifiers.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing the dataset records.
        transform (callable, optional): Optional transform to be applied on an image sample.
        target_label (str): The column name to be used as the primary target label.
        split (str): Indicates whether to use 'train' or 'test' split.
        num_classes_1 (int): Number of classes for tissue density.
        num_classes_2 (int): Number of classes for BIRADS rating.
        num_classes_3 (int): Number of classes for a binary outcome.
    """
    
    def __init__(self, transform=None, target_label='tissueden',
                 split='train', test_size=0.15, val_size=0.15, random_state=42):
        """
        Initialize the MammographyDataset_complete.
        
        Parameters:
            transform (callable, optional): Transformation function to apply to each image.
            target_label (str): Column name in the CSV to be used as the label for training.
            split (str): Specifies which split of the dataset to use ('train' or 'test').
            test_size (float): Fraction of data to use for testing (used in splitting logic).
            val_size (float): Fraction of data to use for validation (used in splitting logic).
            random_state (int): Random seed for reproducibility.
        """
        file_path = Path("processed_data.csv") # Please reach out to us for processed master csv, as it will take a lot of time to clean data yourself.
        if file_path.exists():
            df = pd.read_csv(file_path)
        else:
            print("File does not exist")
        
        # Define the number of classes for different targets
        self.num_classes_1 = 4
        self.num_classes_2 = 3
        self.num_classes_3 = 2
        
        # Filter dataframe to include only valid tissue density values
        df = df[df['tissueden'].isin([0, 1, 2, 3])]
        
        # Filter to include only valid assessment values
        valid_asses_values = ['N', 'B', 'P', 'A', 'S', 'M', 'K']
        df = df[df['asses'].isin(valid_asses_values)]
        
        # Map assessment values to BIRADS classes using a custom mapping
        class_mapping = {
            'N': 1,  # BIRADS 1
            'B': 2,  # BIRADS 2
            'P': 2,  # BIRADS 3
            'A': -1,  # BIRADS 0 (This means need additional evidence, thus treated as incomplete label)
            'S': 3,  # BIRADS 4
            'M': 3,  # BIRADS 5
            'K': 3   # BIRADS 6
        }
        df['birads'] = df['asses'].map(class_mapping)
        # Drop rows where the mapping failed (i.e., resulted in NaN)
        df = df.dropna(subset=['birads'])
        # Convert birads column to integer and adjust to be zero-indexed
        df['birads'] = df['birads'].astype(int)
        df['birads'] = df['birads'] - 1
        
        # Set the cancer risk label for 1-year risk using the helper function
        df = set_cancer_risk_label(df, 1)

        # Select records based on the dataset split
        if split == 'test':
            chosen_ids = df[df['split'] == 'test']['record_id']
        else:
            chosen_ids = df[df['split'] == 'train']['record_id']
        df = df[df['record_id'].isin(chosen_ids)]
        
        # Columns containing the image file paths for different views
        cols = ['c_view_cc', 'c_view_mlo', '2d_cc', '2d_mlo']

        # Reset the dataframe index and assign attributes
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_label = target_label
        self.split = split
        self.print_label_distribution()

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve the sample corresponding to the given index.
        
        Returns:
            tuple: A tuple containing:
                - images (tuple): Transformed images from the four views (c_view_cc, c_view_mlo, 2d_cc, 2d_mlo).
                - labels (tuple): Labels corresponding to tissue density, cancer risk, and BIRADS.
                - record_id: Unique identifier for the record.
        """
        row = self.df.iloc[idx]

        # Extract file paths for each image view
        c1 = row['c_view_cc']
        c2 = row['c_view_mlo']
        c3 = row['2d_cc']
        c4 = row['2d_mlo']

        def load_img(path):
            """
            Load an image from the given path, applying transformations if provided.
            
            Parameters:
                path (str): The file path to the image.
                
            Returns:
                Tensor or PIL Image: The transformed image if a transform is provided,
                                     otherwise the image in PIL format. Returns a zero tensor
                                     if the image is missing or the file does not exist.
            """
            # If path is missing or file does not exist, return a zero tensor with the correct shape
            if pd.isna(path) or not os.path.exists(path):
                miss_data = torch.zeros((3, 224, 224))
                print("!!!WARNING : PLEASE DOUBLE CHECK YOUR DATA FOLDER, THERE IS DATA MISSING!!!")
                return miss_data
            else:
                # Replace file extension from dcm to jpg and apply transformation if available
                image = Image.open(path.replace("dcm", "jpg"))
                if self.transform:
                    return self.transform(image)
                else:
                    return image

        # Load images for each view using the helper function
        c_view_cc = load_img(c1)
        c_view_mlo = load_img(c2)
        d2_cc = load_img(c3)
        d2_mlo = load_img(c4)
        
        # Extract labels from the dataframe row
        label_1 = row["tissueden"]
        label_2 = row["birads"]
        label_21 = row["cancer_risk_label1"]
        
        # Get the record identifier
        record_id = row['record_id']

        return (c_view_cc, c_view_mlo, d2_cc, d2_mlo), (label_1, label_21, label_2), record_id

    def print_label_distribution(self):
        """
        Print the distribution of labels for the selected target.
        
        This function calculates and prints the percentage distribution of each label
        in the dataset based on the provided target label.
        """
        if self.target_label == 'tissueden':
            label_counts = self.df['tissueden'].value_counts()
        else:
            label_counts = self.df[self.target_label].value_counts()
        total = label_counts.sum()
        print(f"{self.target_label} Label Distribution:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} ({(count/total)*100:.2f}%)")

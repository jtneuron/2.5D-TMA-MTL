import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random


def process_dataset(feature_path, boundary_data_path, tumor_data_path, save_root, save_name):
    """
    Process and save dataset features for tumor classification
    
    Args:
        feature_path (str): Path to file containing patient IDs and labels
        boundary_data_path (str): Path to directory containing boundary feature files
        tumor_data_path (str): Path to directory containing tumor feature files  
        save_root (str): Directory to save processed dataset
        
    Returns: None

    """
    datas = []
    processed_data = []
    processed_data_path = os.path.join(save_root, save_name)
        
    with open(feature_path, 'r') as f:
        for line in f:
            datas.append(line.strip())
            
    feature_list = os.listdir(boundary_data_path)
    features = [os.path.join(boundary_data_path, f) for f in feature_list]
    
    for data in tqdm(datas, desc='Processing dataset', total=len(datas)):
        try:        
            data_dict = {}
            boundary_features, tumor_features = [], []
            patient_id = data.split(',')[0]
            for feature_pt in features:
                if patient_id in feature_pt.split('/')[-1].split('.')[0]:
                    feature_name = feature_pt.split('/')[-1].split('.')[0]
                    boundary_feature = torch.load(feature_pt)
                    boundary_features.append((feature_name, boundary_feature.unsqueeze(0)))
                    tumor_feature = torch.load(os.path.join(tumor_data_path, feature_name+'.pt'))
                    tumor_features.append((feature_name, tumor_feature.unsqueeze(0)))
            data_dict['patient_id'] = patient_id
            data_dict['boundary_features'] = boundary_features
            data_dict['tumor_features'] = tumor_features
            data_dict['label'] = int(data.split(',')[1])
            processed_data.append(data_dict)
        except:
            print(data)
    torch.save(processed_data, processed_data_path)
    return 


def process_dataset_survival(feature_path, boundary_data_path, tumor_data_path, save_root, save_name):
    """
    Process and save dataset features for tumor classification
    
    Args:
        feature_path (str): Path to file containing patient IDs and labels
        boundary_data_path (str): Path to directory containing boundary feature files
        tumor_data_path (str): Path to directory containing tumor feature files  
        save_root (str): Directory to save processed dataset
        
    Returns: None

    """
    datas = []
    processed_data = []
    os.makedirs(save_root, exist_ok=True)
    processed_data_path = os.path.join(save_root, save_name)
        
    with open(feature_path, 'r') as f:
        for line in f:
            datas.append(line.strip())
            
    feature_list = os.listdir(boundary_data_path)
    features = [os.path.join(boundary_data_path, f) for f in feature_list]
    
    for data in tqdm(datas, desc='Processing dataset', total=len(datas)):
        try:        
            data_dict = {}
            boundary_features, tumor_features = [], []
            patient_id = data.split(',')[0]
            extracted_data = data.split(',')
            for feature_pt in features:
                if patient_id in feature_pt.split('/')[-1].split('.')[0]:
                    feature_name = feature_pt.split('/')[-1].split('.')[0]
                    boundary_feature = torch.load(feature_pt)
                    boundary_features.append((feature_name, boundary_feature))
                    tumor_feature = torch.load(os.path.join(tumor_data_path, feature_name+'.pt'))
                    tumor_features.append((feature_name, tumor_feature))
            data_dict['patient_id'] = patient_id
            data_dict['boundary_features'] = boundary_features
            data_dict['tumor_features'] = tumor_features
            data_dict['lauren_label'] = int(extracted_data[1])
            data_dict['CPS_label'] = int(extracted_data[2])
            data_dict['her2_label'] = int(extracted_data[3])
            data_dict['mmr_label'] = int(extracted_data[4])
            data_dict['Clauding_label'] = int(extracted_data[5])
            data_dict['recurrence_status'] = int(extracted_data[6])
            data_dict['recurrence_time'] = float(extracted_data[7])
            data_dict['survival_status'] = int(extracted_data[8])
            data_dict['survival_time'] = float(extracted_data[9])
            processed_data.append(data_dict)
        except:
            print(data)
    torch.save(processed_data, processed_data_path)
    return 
        

def process_dataset_2D(txt_path, feature_data_path, save_root, save_name):
    """
    Process and save dataset features for tumor classification
    
    Args:
        feature_path (str): Path to file containing patient IDs and labels
        boundary_data_path (str): Path to directory containing boundary feature files
        tumor_data_path (str): Path to directory containing tumor feature files  
        save_root (str): Directory to save processed dataset
        
    Returns: None

    """
    datas = []
    processed_data = []
    os.makedirs(save_root, exist_ok=True)
    processed_data_path = os.path.join(save_root, save_name)
        
    with open(txt_path, 'r') as f:
        for line in f:
            datas.append(line.strip())
            
    feature_list = os.listdir(feature_data_path)
    features = [os.path.join(feature_data_path, f) for f in feature_list]
    
    for data in tqdm(datas, desc='Processing dataset', total=len(datas)):
        try:        
            data_dict = {}
            patient_id = data.split(',')[0]
            extracted_data = data.split(',')
            for feature_pt in features:
                if str('_') in feature_pt.split('/')[-1]:
                    if patient_id == feature_pt.split('/')[-1].split('_')[0]:
                        feature = torch.load(feature_pt)
                        break
                else:
                    if patient_id == feature_pt.split('/')[-1].split('.')[0]:
                        feature = torch.load(feature_pt)
                        break
            data_dict['patient_id'] = patient_id
            data_dict['features'] = feature
            data_dict['lauren_label'] = int(extracted_data[1])
            data_dict['CPS_label'] = int(extracted_data[2])
            data_dict['her2_label'] = int(extracted_data[3])
            data_dict['mmr_label'] = int(extracted_data[4])
            data_dict['Clauding_label'] = int(extracted_data[5])
            data_dict['recurrence_status'] = int(extracted_data[6])
            data_dict['recurrence_time'] = float(extracted_data[7])
            data_dict['survival_status'] = int(extracted_data[8])
            data_dict['survival_time'] = float(extracted_data[9])
            processed_data.append(data_dict)
        except:
            print(data)
    torch.save(processed_data, processed_data_path)
    return 




def write_txt(data, save_path):
    """
    Write data to txt file
    
    Args:
        data (list): List of data
        save_path (str): Path to save txt file
        
    Returns: None
    """
    with open(save_path, 'w') as f:
        for item in data:
            f.write(item + '\n')
    return


def generate_single_dataset(feature_path, data_path, train_save_path, test_save_path, train_ratio=0.7):
    """
    Generate single dataset
    
    Args:
        data_path (str): Path to dataset file
        save_path (str): Path to save txt file
        
    Returns: None

    """
    datas = []
    data_dict = {}
    with open(feature_path, 'r') as f:
        for line in f:
            datas.append(line.split(',')[0].strip())
            data_dict[line.split(',')[0].strip()] = line.split(',')[1].strip()

    random.shuffle(datas)
    train_size = int(len(datas) * train_ratio)
    train_data_list = datas[:train_size]
    test_data_list = datas[train_size:]
    
    feature_list = os.listdir(data_path)
    features = [os.path.join(data_path, f) for f in feature_list]

    train_data = []
    test_data = []
    
    for feature in tqdm(features, desc='Processing dataset', total=len(features)):
        name = feature.split('/')[-1].split('-',2)
        feature_name = name[0]+'-'+name[1]
        if feature_name in train_data_list:
            train_data.append(feature+','+str(int(data_dict[feature_name])))
        elif feature_name in test_data_list:
            test_data.append(feature+','+str(int(data_dict[feature_name])))

    random.shuffle(train_data)
    random.shuffle(test_data)
    write_txt(train_data, train_save_path)
    write_txt(test_data, test_save_path)

    return

def del_cls(file_path, save_path, flag):
    """
    Filter and save lines from a file that match a specific class flag
    
    Args:
        file_path (str): Path to input file containing comma-separated data
        save_path (str): Path to save filtered output file
        flag (str): Class label to filter by (last element in each line)
        
    Returns:
        None: Writes filtered lines to save_path
    """
    save_list = []
    with open(file_path) as f:
        for line in f.readlines():
            if line.split(',')[-1].split('\n')[0] != str(flag):
                save_list.append(line.split('\n')[0])
    write_txt(save_list, save_path)
    return

def del_cls_survival(file_path, save_path, flag):
    """
    Filter and save lines from a file that match a specific class flag
    
    Args:
        file_path (str): Path to input file containing comma-separated data
        save_path (str): Path to save filtered output file
        flag (str): Class label to filter by (last element in each line)
        
    Returns:
        None: Writes filtered lines to save_path
    """
    save_list = []
    with open(file_path) as f:
        for line in f.readlines():
            if line.split(',')[1] != str(flag):
                save_list.append(line.split('\n')[0])
    write_txt(save_list, save_path)
    return



def load_lauren_dataset(data_path='/traindata/lwt_project/Path2D5/dataset_pth/conch_lauren_data.pt', train_ratio=0.8):
    """
    Load Lauren dataset and split into train/test sets
    
    Args:
        data_path (str): Path to dataset file
        train_ratio (float): Ratio of training data, default 0.8
        
    Returns:
        train_data (list): Training dataset
        test_data (list): Testing dataset
    """
    # Load data
    data = torch.load(data_path)
    
    # Shuffle data
    random.shuffle(data)
    
    # Split into train/test
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    return train_data, test_data


def single_lauren_dataset(train_data_path, test_data_path):
    """
    Load Lauren dataset and split into train/test sets
    
    Args:
        data_path (str): Path to dataset file
        train_ratio (float): Ratio of training data, default 0.8
        
    Returns:
        train_data (list): Training dataset
        test_data (list): Testing dataset
    """
    # Load data
    train_data = []
    with open(train_data_path, 'r') as f1:
        for line1 in f1:
            train_data.append(line1.strip())
    test_data = []
    with open(test_data_path, 'r') as f2:
        for line2 in f2:
            test_data.append(line2.strip())
    
    
    print(f"Total samples: {len(train_data)+len(test_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    return train_data, test_data




class CustomDataset(Dataset):
    def __init__(self, data):
        self.samples = data
           
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        id = sample['patient_id']
        boundary_feature = []
        if sample['boundary_features'] is not None:
            for feature in sample['boundary_features']:
                boundary_feature.append(feature[1])
        tumor_feature = []
        for feature in sample['tumor_features']:
            tumor_feature.append(feature[1])
        label = dict(list(sample.items())[3:])

        return id, boundary_feature, tumor_feature, label
    

class CustomDataset_2D(Dataset):
    def __init__(self, data):
        self.samples = data
           
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        id = sample['patient_id']
        feature = sample['features']
        label = dict(list(sample.items())[2:])

        return id, feature, label    
    

class SurvivalDataset(Dataset):
    def __init__(self, data):
        self.samples = data
           
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = dict()
        id = sample['patient_id']
        boundary_feature = []
        for feature in sample['boundary_features']:
            boundary_feature.append(feature[1])
        tumor_feature = []
        for feature in sample['tumor_features']:
            tumor_feature.append(feature[1])
        label['lauren_label'] = sample['lauren_label']
        label['recurrence_status'] = sample['recurrence_status']
        label['recurrence_time'] = sample['recurrence_time']
        label['survival_status'] = sample['survival_status']
        label['survival_time'] = sample['survival_time']

        return id, boundary_feature, tumor_feature, label


class Single_CustomDataset(Dataset):
    def __init__(self, data):
        self.samples = data
           
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample.split(',')[1]
        feature = torch.load(sample.split(',')[0])

        return feature, int(label)

class Single_SurvivalDataset(Dataset):
    def __init__(self, data):
        self.samples = data
           
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        label = dict()
        sample = self.samples[idx]
        data = sample.split(',')
        feature = torch.load(data[0])
        label['lauren_label'] = int(data[1])
        label['recurrence_status'] = int(data[2])
        label['recurrence_time'] = float(data[3])
        label['survival_status'] = int(data[4])
        label['survival_time'] = float(data[5])

        return feature, label


def generate_xiamen_dataset(data_path, save_path):

    file_list = os.listdir(data_path)
    for file in file_list:
        file_name = file.split('_')
        depth_id = file_name[2]
        file_paths = os.listdir(os.path.join(data_path, file))
        for file_path in tqdm(file_paths, desc='数据处理', ncols=70):
            file_root = os.path.join(data_path, file)
            pt = torch.load(os.path.join(file_root, file_path))
            file_name_ = file_path.split('_')[0].split('-')
            if int(file_name_[1])<10:
                file_name_[1] = '0'+file_name_[1]
            new_name = 'TMA1-'+file_name_[0]+file_name_[1]+'-'+depth_id+'.pt'
            torch.save(pt, os.path.join(save_path, new_name))    
    return 




def process_xiamen_dataset(feature_path, tumor_data_path, save_root, save_name):
    """
    Process and save dataset features for tumor classification

    Args:
        feature_path (str): Path to file containing patient IDs and labels
        boundary_data_path (str): Path to directory containing boundary feature files
        tumor_data_path (str): Path to directory containing tumor feature files  
        save_root (str): Directory to save processed dataset
        
    Returns: None

    """
    datas = []
    processed_data = []
    os.makedirs(save_root, exist_ok=True)
    processed_data_path = os.path.join(save_root, save_name)
        
    with open(feature_path, 'r') as f:
        for line in f:
            datas.append(line.strip())
            
    feature_list = os.listdir(tumor_data_path)
    features = [os.path.join(tumor_data_path, f) for f in feature_list]

    for data in tqdm(datas, desc='Processing dataset', total=len(datas)):
        try:        
            data_dict = {}
            tumor_features = []
            patient_id = data.split(',')[0]
            extracted_data = data.split(',')
            for feature_pt in features:
                if patient_id in feature_pt.split('/')[-1].split('.')[0]:
                    feature_name = feature_pt.split('/')[-1].split('.')[0]
                    tumor_feature = torch.load(os.path.join(tumor_data_path, feature_name+'.pt'))
                    tumor_features.append((feature_name, tumor_feature))
            data_dict['patient_id'] = patient_id
            data_dict['boundary_features'] = None
            data_dict['tumor_features'] = tumor_features
            data_dict['lauren_label'] = int(extracted_data[1])
            data_dict['recurrence_status'] = int(extracted_data[2])
            data_dict['recurrence_time'] = float(extracted_data[3])
            data_dict['survival_status'] = int(extracted_data[4])
            data_dict['survival_time'] = float(extracted_data[5])
            processed_data.append(data_dict)
        except:
            print(data)
    torch.save(processed_data, processed_data_path)
    return 




        


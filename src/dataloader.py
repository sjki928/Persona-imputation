import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


STD_COL=['GENDER',
        'AGE_GRP',
        'TRAVEL_STYL_1',
        'TRAVEL_STYL_2',
        'TRAVEL_STYL_3',
        'TRAVEL_STYL_4',
        'TRAVEL_STYL_5',
        'TRAVEL_STYL_6',
        'TRAVEL_STYL_7',
        'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1',
        'TRAVEL_COMPANIONS_NUM',
        'PAYMENT_NORMAL']

STD_DICT = { k: [0,1] if k == 'GENDER' else [0,7] for k in STD_COL}

def build_dataframe(csv_path,embedding_path,test_size=0.2):
    
    dataset = pd.read_csv(csv_path)
    
    scaler = StandardScaler()
    
    train_name ,test_name = train_test_split(dataset.VISIT_AREA_NM.unique(),test_size=test_size)
    
    train_mask = dataset.VISIT_AREA_NM.isin(train_name)
    test_mask = dataset.VISIT_AREA_NM.isin(test_name)
    
    train = dataset[train_mask]
    test = dataset[test_mask]

    scaler.fit(train.loc[:,STD_COL])
    
    train.loc[:,STD_COL] = scaler.transform(train.loc[:,STD_COL])
    test.loc[:,STD_COL] = scaler.transform(test.loc[:,STD_COL])

    def _add_embedding(dataset,embedding_path):
        
        dataframe = dataset.loc[:,['VISIT_AREA_NM']+STD_COL].groupby('VISIT_AREA_NM').mean().fillna(0)
        dataframe_var = dataset.loc[:,['VISIT_AREA_NM']+STD_COL].groupby('VISIT_AREA_NM').var().fillna(0)
        
        dataframe['mean_'] = dataframe.apply(lambda x: [x[col] for col in STD_COL],axis=1)
        dataframe['var_'] = dataframe_var.apply(lambda x: [x[col] +1e-6 for col in STD_COL],axis=1)
        dataframe = dataframe[['mean_','var_']].reset_index()
        
        dataframe = pd.merge(dataframe,dataset.loc[:,['VISIT_AREA_NM','overview']].drop_duplicates(),how='left',on='VISIT_AREA_NM')
        
        li = dataframe.overview.tolist()
        model = SentenceTransformer(embedding_path)
        embeddings = model.encode(li)
        dataframe['textvec'] = list(embeddings)
        
        return dataframe
    
    train = _add_embedding(train,embedding_path)
    test = _add_embedding(test,embedding_path)
    
    return train, test, scaler


class TourDataset(Dataset):
    
    def __init__(self, 
                 dataframe,
                 ):
        
        super(TourDataset).__init__()
        self.visit_area_nm = dataframe["VISIT_AREA_NM"]
        self.mean = dataframe["mean_"]
        self.var = dataframe["var_"]
        self.overview = dataframe["overview"]
        self.textvec = dataframe["textvec"]
        

    def __len__(self):
        
        return len(self.var)

    def __getitem__(self, idx):
        visit_area_nm = self.visit_area_nm.iloc[idx]
        mean = torch.Tensor(self.mean.iloc[idx])
        var = torch.Tensor(self.var.iloc[idx])
        overview = self.overview.iloc[idx]
        textvec = self.textvec.iloc[idx]
        
        return {
            'visit_area_nm' : visit_area_nm,
            'mean' : mean,
            'var' : var,
            'overview' : overview,
            'textvec' : textvec
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataframe, test_dataframe, scaler = build_dataframe('/workspace/vae_train_nara.csv','sentence-transformers/all-MiniLM-L6-v2')
    
    train_dataset = TourDataset(train_dataframe)
    test_dataset = TourDataset(test_dataframe)
    
    train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)
    batch = next(iter(train_dataloader))
    print(1)
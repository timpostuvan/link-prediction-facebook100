import pandas as pd 

train = pd.read_csv('./node_based_train.data',sep='\t')
test = pd.read_csv('./node_based_test.data',sep='\t')
train[['is_dorm','is_year','label']].to_csv('./node_based_train_filtered.data',index=False)
test[['is_dorm','is_year','label']].to_csv('./node_based_test_filtered.data',index=False)


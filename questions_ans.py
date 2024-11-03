import pandas as pd

# Load JSON from a file
df_q = pd.read_json('v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json')
df_a = pd.read_json('v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json')

print(df_a.unique())
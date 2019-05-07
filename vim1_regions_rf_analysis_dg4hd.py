### Computational Neuroscience fMRI Random Forests Analysis ######

#%% Define Dataset Path and Import Dependencies
response_path = "C:/Users/Student/comp-neurosci/EstimatedResponses.mat"
stimuli_path = "C:/Users/Student/comp-neurosci/Stimuli.mat"

#Import Dependencies
import pandas as pd
import tables,numpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import random
random.seed(3456)

#%% Load in Data

#Load in Response

# Open specific task (Trn/Val), subject, and region, with input specified. Returns a Dataframe where columns
# correspond to each voxel in a cortical region, and rows correspond to each stimulus image
def get_response_task_subj_reg(task,subj,region):
    f = tables.open_file(response_path)
    if task not in ['Trn','Val']:
        print("Task name not recognized")
        return
    if subj not in [1,2]:
        print("Subject number not recognized")
        return
    if region not in range(0,8):
        print("Subject number not recognized")
        return   
    Dat = f.get_node('/data'+str(task)+'S'+str(subj))[:] 
    ROI = f.get_node('/roiS'+str(subj))[:].flatten() 
    idx = numpy.nonzero(ROI==region)[0] 
    resp = pd.DataFrame(Dat[:,idx])
    resp.columns = idx
    return resp


#Load in Stimuli

# Get stimuli for each type of task (Trn/Val). Returns a list of DataFrames, where each DataFrame is an image
def get_stimuli_images(task):
    s = scipy.io.loadmat(stimuli_path)
    images = s['stim'+str(task)]
    list_images_dfs = []
    for i in range(0,len(images)):
        list_images_dfs.append(pd.DataFrame(images[i]))
    return list_images_dfs  

#%% Select Training Set Data

#Select training set images that correspond to "People" and "Animals": the indices for the arrays above 
# were mannually determined for the first 316 images
    
images_trn = get_stimuli_images('Trn')

ppl_trn = np.array([ 2, 19, 20, 21, 24, 25, 26, 30, 32, 40, 
                51, 54, 62, 83, 88, 109, 121, 123, 125, 126,
                131, 132, 133, 138, 144, 152, 167, 169, 194, 213,
                219, 230, 259, 268, 280, 288, 290, 292, 302, 310,
                311, 312, 316])
ani_trn = np.array([ 3,  5,  6,  7,  8, 13, 17, 18, 23, 27, 
                34, 36, 42, 46, 52, 56, 59, 63, 66, 77,
                86, 97, 99, 100, 103, 106, 110, 116, 122, 136,
                140, 147, 148, 151, 155, 156, 161, 165, 177, 180,
                186, 188, 193, 200, 204, 210, 211, 212, 224, 226, 
                228, 229, 232, 233, 236, 238, 242, 243, 252, 253,
                254, 255, 261, 262, 263, 264, 276, 282, 284, 301,
                303, 305, 306, 309, 314]) 
img_people = []
img_animal = []
for i in ppl_trn:
    img_people.append(images_trn[i])
for j in ani_trn:
    img_animal.append(images_trn[j])
    
#Get sets of training responses for each of V1, V2, and V3 for subject 1
all_responses = []
subj = 1
for i in range(1,4):
    responses={}
    for task in ['Trn']:
        for region in range(i,i+1):
            responses["resp{0}{1}/{2}".format(task, subj, region)]=get_response_task_subj_reg(task,subj,region)
    all_responses.append(responses)
v1_trn_resp = all_responses[0]
v2_trn_resp = all_responses[1]
v3_trn_resp = all_responses[2]

#Extract rows from the above responses corresponding to images classified as People and Animals, 
# label them accordingly and concatonate them into a new combined dataframe

def create_training_set(resp):
    comb_df = pd.DataFrame()
    resp = resp[list(resp.keys())[0]]
    comb_df = pd.concat([resp.loc[ppl_trn], comb_df], axis = 0)
    comb_df = pd.concat([resp.loc[ani_trn], comb_df], axis = 0)
    comb_df['Category'] = 'Animal'
    comb_df.loc[ppl_trn,'Category'] = 'People'
    comb_df = comb_df.fillna(0)
    return comb_df

comb_df_v1 = create_training_set(v1_trn_resp)
comb_df_v2 = create_training_set(v2_trn_resp)
comb_df_v3 = create_training_set(v3_trn_resp)

#%% Select Testing Set Data

#Select all training set images that correspond to "People" and "Animals" 
images_val = get_stimuli_images('Val')

ppl_val = np.array([ 1, 7, 20, 21, 25, 26, 33, 40, 47, 
                52, 57, 71, 72, 73, 77, 79, 80, 85, 86, 
                90, 94, 112])
ani_val = np.array([ 3, 6, 9, 15, 16, 22, 29, 30, 32, 36, 
                37, 39, 41, 43, 48, 49, 51, 59, 66, 75,
                76, 78, 83, 89, 92, 98, 109, 110, 111, 116,
                117, 118]) 
img_people = []
img_animal = []
for i in ppl_val:
    img_people.append(images_val[i])
for j in ani_val:
    img_animal.append(images_val[j])

#Get sets of training responses for each of V1, V2, and V3 for both subjects 1 and 2
all_responses = []
subj = 1
for i in range(1,4):
    responses={}
    for task in ['Trn']:
            for region in range(i,i+1):
                responses["resp{0}{1}/{2}".format(task, subj, region)]=get_response_task_subj_reg(task,subj,region)
    all_responses.append(responses)
v1_val_resp = all_responses[0]
v2_val_resp = all_responses[1]
v3_val_resp = all_responses[2]

#Extract rows corresponding to images classified as People and Animals and 
# label them accordingly

def create_test_set(resp):
    comb_df_test = pd.DataFrame()
    resp = resp[list(resp.keys())[0]]
    comb_df_test = pd.concat([resp.loc[ppl_val], comb_df_test], axis = 0)
    comb_df_test = pd.concat([resp.loc[ani_val], comb_df_test], axis = 0)
    comb_df_test['Category'] = 'Animal'
    comb_df_test.loc[ppl_val,'Category'] = 'People'
    comb_df_test = comb_df_test.fillna(0)
    return comb_df_test

comb_df_test_v1 = create_test_set(v1_val_resp)
comb_df_test_v2 = create_test_set(v2_val_resp)
comb_df_test_v3 = create_test_set(v3_val_resp)

#%% Build Random Forest Model on Training Set Data by Splitting Training Set into Train-Test

def model_build(comb_df):
    x_train, x_test, y_train, y_test = train_test_split(comb_df.iloc[:,:-1], comb_df.iloc[:,-1], test_size=0.4)
    rfc=RandomForestClassifier(n_estimators=100)
    model = rfc.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    feature_imp = pd.Series(model.feature_importances_,index= x_train.columns).sort_index()
    return model, y_pred, acc, feature_imp
   
model_v1, pred_v1, acc_v1, feat_imp_v1 = model_build(comb_df_v1)
model_v2, pred_v2, acc_v2, feat_imp_v2 = model_build(comb_df_v2)
model_v3, pred_v3, acc_v3, feat_imp_v3 = model_build(comb_df_v3)


#%% Optimize Models by Feature Selection
# The for loop in the function below serves as the feature selection algorithm: predictors with lowest
# importance are sequentially removed and models are re-trained on the remaining. The model with the best 
# accuracy score is the output

def model_optimization(comb_df):
    x_train, x_test, y_train, y_test = train_test_split(comb_df.iloc[:,:-1], comb_df.iloc[:,-1], test_size=0.4)
    rfc=RandomForestClassifier(n_estimators=100)
    model = rfc.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    feature_imp = pd.Series(model.feature_importances_,index= x_train.columns).sort_values()
    feat_remov = [0]
    accur = [acc]
    for i in range(1, len(feature_imp)-1):
        select = feature_imp.index[i:-1].values.tolist()
        features = x_train[select]
        model = rfc.fit(features, y_train)
        y_pred=model.predict(x_test[select])
        acc = metrics.accuracy_score(y_test, y_pred)
        feat_remov.append(i)
        accur.append(acc)
    max_acc = max(accur)
    indices = [i for i, x in enumerate(accur) if x == max_acc]
    n_feat = [ feat_remov[i] for i in indices]
    select_opt = feature_imp.index[max(n_feat):-1].values.tolist()
    features_opt = x_train[select_opt]
    model_opt = rfc.fit(features_opt, y_train)
    y_pred_opt = model.predict(x_test[select_opt])
    acc_opt = metrics.accuracy_score(y_test, y_pred_opt)
    return model_opt, y_pred_opt, acc_opt

model_opt_1, y_pred_opt_1, acc_opt_1 = model_optimization(comb_df_v1)
model_opt_2, y_pred_opt_2, acc_opt_2 = model_optimization(comb_df_v2)
model_opt_3, y_pred_opt_3, acc_opt_3 = model_optimization(comb_df_v3)


#%% Retrain Model with Full Training Set & Optimized Parameters, and Test on Validated Set

def model_final(comb_df, comb_test_df):
    x_train = comb_df.iloc[:,:-1]
    y_train = comb_df.iloc[:,-1]
    x_test = comb_test_df.iloc[:,:-1]
    y_test = comb_test_df.iloc[:,-1]
    rfc=RandomForestClassifier(n_estimators=100)
    model = rfc.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    feature_imp = pd.Series(model.feature_importances_,index= x_train.columns).sort_values()
    feat_remov = [0]
    accur = [acc]
    for i in range(1, len(feature_imp)-1):
        select = feature_imp.index[i:-1].values.tolist()
        features = x_train[select]
        model = rfc.fit(features, y_train)
        y_pred=model.predict(x_test[select])
        acc = metrics.accuracy_score(y_test, y_pred)
        feat_remov.append(i)
        accur.append(acc)
    max_acc = max(accur)
    indices = [i for i, x in enumerate(accur) if x == max_acc]
    n_feat = [ feat_remov[i] for i in indices]
    select_opt = feature_imp.index[max(n_feat):-1].values.tolist()
    features_opt = x_train[select_opt]
    model_opt = rfc.fit(features_opt, y_train)
    y_pred_opt = model.predict(x_test[select_opt])
    acc_opt = metrics.accuracy_score(y_test, y_pred_opt)
    feature_imp_opt = pd.Series(model.feature_importances_,index= features_opt.columns).sort_index()
    return model_opt, y_pred_opt, acc_opt, feature_imp_opt

model_final_v1, y_pred_v1, acc_final_v1, feat_opt_v1 = model_final(comb_df_v1, comb_df_test_v1)
model_final_v2, y_pred_v2, acc_final_v2, feat_opt_v2 = model_final(comb_df_v2, comb_df_test_v2)
model_final_v3, y_pred_v3, acc_final_v3, feat_opt_v3 = model_final(comb_df_v3, comb_df_test_v3)

#%% 3-D Models of Voxel and Importance
# The unravel_index function converts the 1-dimensional index from the model features to a 3-dimensional
#index

plt.rcdefaults()
fig = plt.figure(figsize=(30, 20))

index_v1 = np.array(feat_opt_v1.index.astype(int).sort_values())
idx_1,idy_1,idz_1 = np.unravel_index(index_v1,(64,64,18))

index_v2 = np.array(feat_opt_v2.index.astype(int).sort_values())
idx_2,idy_2,idz_2 = np.unravel_index(index_v2,(64,64,18))

index_v3 = np.array(feat_opt_v3.index.astype(int).sort_values())
idx_3,idy_3,idz_3 = np.unravel_index(index_v3,(64,64,18))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.scatter(idx_1, idy_1, idz_1, zdir='z', s=30, c='b', depthshade=True)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.scatter(idx_2, idy_2, idz_2, zdir='z', s=30, c='r', depthshade=True)
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.scatter(idx_3, idy_3, idz_3, zdir='z', s=30, c='g', depthshade=True)

index_v1 = np.array(comb_df_v1.columns[0:-1].astype(int).sort_values())
idx_1,idy_1,idz_1 = np.unravel_index(index_v1,(64,64,18))

index_v2 = np.array(comb_df_v2.columns[0:-1].astype(int).sort_values())
idx_2,idy_2,idz_2 = np.unravel_index(index_v2,(64,64,18))

index_v3 = np.array(comb_df_v3.columns[0:-1].astype(int).sort_values())
idx_3,idy_3,idz_3 = np.unravel_index(index_v3,(64,64,18))

ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.scatter(idx_1, idy_1, idz_1, zdir='z', s=30, c='b', depthshade=True)
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.scatter(idx_2, idy_2, idz_2, zdir='z', s=30, c='r', depthshade=True)
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.scatter(idx_3, idy_3, idz_3, zdir='z', s=30, c='g', depthshade=True)

ax1.set_xlabel('$X$', fontsize = 20)
ax1.set_ylabel('$Y$', fontsize = 20)
ax1.set_zlabel('$Z$', fontsize = 20)
ax1.set_title('V1 Selected Voxels', fontsize = 35)
ax2.set_title('V2 Selected Voxels', fontsize = 35)
ax3.set_title('V3 Selected Voxels', fontsize = 35)
ax4.set_title('V1 All Voxels', fontsize = 35)
ax5.set_title('V2 All Voxels', fontsize = 35)
ax6.set_title('V3 All Voxels', fontsize = 35)

plt.show()


#%% Plot Accuracy for Different Predictions 

#Build dataframe with appropriate values
accuracy_df = pd.DataFrame({'Model':['Initial Build','Optimized Build','Final Build','Initial Build','Optimized Build','Final Build','Initial Build','Optimized Build','Final Build'],
                            'Region': ['V1','V1','V1','V2','V2','V2','V3','V3','V3'],
                            'Accuracy':[acc_v1,acc_opt_1,acc_final_v1,acc_v2,acc_opt_2,acc_final_v2,acc_v3,acc_opt_3,acc_final_v3]})

# set width of bar
barWidth = 0.25
 
# set height of bar
init_build = [acc_v1, acc_v2, acc_v3]
opt_build = [acc_opt_1,acc_opt_2,acc_opt_3]
final_build = [acc_final_v1,acc_final_v2,acc_final_v3]
 
# Set position of bar on X axis
r1 = np.arange(len(init_build))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
   
# Make the plot
plt.bar(r1, init_build, color='#7f6d5f', width=barWidth, edgecolor='white', label='Initial Build')
plt.bar(r2, opt_build, color='#557f2d', width=barWidth, edgecolor='white', label='Optimized Build')
plt.bar(r3, final_build, color='#2d7f5e', width=barWidth, edgecolor='white', label='Final Build')
 
# Add xticks on the middle of the group bars
plt.xlabel('Region', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Model Accuracy by Region')
plt.xticks([r + barWidth for r in range(len(init_build))], ['V1', 'V2', 'V3'])
 
# Create legend & Show graphic
plt.legend()
plt.show()


#%% Generate Example Images

fig=plt.figure(figsize=(10, 10))
columns = 2
rows = 2
for i in range(1, 3):
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_people[i], cmap='gray')
for i in range(3,5):
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_animal[i], cmap='gray')
plt.show()













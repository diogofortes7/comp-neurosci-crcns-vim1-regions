# All data for the project is contained within the 'EstimatedResponses.mat' and 'Stimuli.mat' files. 

Download the EstimatedResponses.mat and Stimuli.mat files from the VIM-1 dataset (https://crcns.org/data-sets/vc/vim-1/about-vim-1). After downloading and saving both files, edit the 'response_path' and 'stimuli_path' variables in the vim1_regions_rf_analysis_dg4hd.py file. The script contains labeled functions to load both a particular estimated response (accodring to subject, ROI, and task) and particular stimuli sets (Training and Validation). The functions are copied below for clarity:

# Load in Data

Load in Response

Open specific task (Trn/Val), subject, and region, with input specified. Returns a Dataframe where columns
correspond to each voxel in a cortical region, and rows correspond to each stimulus image

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


Load in Stimuli

Get stimuli for each type of task (Trn/Val). Returns a list of DataFrames, where each DataFrame is an image

def get_stimuli_images(task):
    s = scipy.io.loadmat(stimuli_path)
    images = s['stim'+str(task)]
    list_images_dfs = []
    for i in range(0,len(images)):
        list_images_dfs.append(pd.DataFrame(images[i]))
    return list_images_dfs  





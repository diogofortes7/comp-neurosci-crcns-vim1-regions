#All data for the project is contained within the 'EstimatedResponses.mat' and 'Stimuli.mat' files. The following functions may be used to input the data into Python

# Open specific task (Trn/Val), subject, and region, with input requested from the user. Returns a Dataframe where columns
# correspond to each voxel in a cortical region, and rows correspond to each stimulus image

def get_response_input():
    f = tables.open_file('EstimatedResponses.mat')
    task = input("Task type (Trn/Val):")
    if task not in ['Trn','Val']:
        print("Task name not recognized")
        return
    subj = int(input("Subject # (1/2):"))
    if subj not in [1,2]:
        print("Subject number not recognized")
        return
    region = int(input("Select region number from list\n[0] other \n[1] V1 \n[2] V2 \n[3] V3 \n[4] V3A \n[5] V3B \n[6] V4 \n[7] LatOcc \n"))
    if region not in range(0,8):
        print("Subject number not recognized")
        return   
    Dat = f.get_node('/data'+str(task)+'S'+str(subj))[:] 
    ROI = f.get_node('/roiS'+str(subj))[:].flatten() 
    idx = numpy.nonzero(ROI==region)[0] 
    resp = Dat[:,idx] 
    plot_pref = input("Plot response (Y/N):")
    if plot_pref == 'Y':
        fig, ax = plt.subplots(figsize=(50, 50))
        plt.imshow(V1resp, cmap='seismic', interpolation='none',aspect='equal')
        plt.show()
    resp = pd.DataFrame(resp)
    return resp
    
# Open specific task (Trn/Val), subject, and region, with input specified. Returns a Dataframe where columns
# correspond to each voxel in a cortical region, and rows correspond to each stimulus image
def get_response_task_subj_reg(task,subj,region):
    f = tables.open_file('EstimatedResponses.mat')
    f.list_nodes
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
    resp = Dat[:,idx] 
    resp = pd.DataFrame(resp)
    return resp
    
# Get groups of responses. Edit the fields in the loops to parse different groups. Final output is a dictionary of 
# DataFrames
responses={}
for task in ['Trn','Val']:
    for subj in [1,2]:
        for region in range(0,8):
            responses["resp{0}{1}/{2}".format(task, subj, region)]=get_response_task_subj_reg(task,subj,region)

# Get stimuli for each type of task (Trn/Val). Returns a list of DataFrames, where each DataFrame is an image
def get_stimuli_images(task):
    s = scipy.io.loadmat('Stimuli.mat')
    images = s['stim'+str(task)]
    list_images_dfs = []
    for i in range(0,len(images)):
        list_images_dfs.append(pd.DataFrame(images[i]))
    return list_images_dfs  

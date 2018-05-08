import numpy as np
####
# 
####

def k_mean(K,d,train_data,max_iteration=100):
    print"--------------------"
    print"d: ",d    
    np.random.seed(0)
    center_idx = np.random.randint(train_data.shape[0], size=K)
    centers=train_data[center_idx,:]
    centers=centers[:,:2]
    MAX_ITERATION=max_iteration
    means=centers.copy()
    variance=np.zeros((means.shape))
    clustered_data=train_data
    clustered_data=np.insert(clustered_data, 2, -1, axis= 1)
    cont=True
    iteration=0
    while(cont):
        iteration+=1
        for idx,data in enumerate(train_data):
            dist=1000
            label=-1
            for i,meand in enumerate(means):
                if np.linalg.norm(data-meand)<dist:
                    dist=np.linalg.norm(data-meand)
                    label=i+1
            clustered_data[idx,2]=label
        old_means=means.copy()
        for i,cluster in enumerate(np.unique(clustered_data[:,2])):
            data_in_cluster=clustered_data[np.where(clustered_data[:,2]==cluster)]
            means[i]=np.mean(data_in_cluster[:,:2],axis=0)
            #variance of each cluster is calculated
            variance[i]=np.var(data_in_cluster[:,:2],axis=0)
        if iteration>MAX_ITERATION or np.allclose(old_means,means):
            cont=False
    
    print "Number of Iterations:",iteration
    
    ######
    ## Select the maximum variance in either x or y direction 
    ## Use the standard deviation from this variance to draw the circles in each cluster
    ######
    
    max_var=np.array(np.ndarray.max(variance,axis=1))[np.newaxis].T
    std_deviation=np.fabs(np.sqrt(max_var))
    plot_data(d,clustered_data,title="Final Clusters",centers=means,std_deviation=std_deviation) 


    return means,max_var


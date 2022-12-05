import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
def  plot_metrics(All_metrics,clf):
    
    plt.figure()
    plt.plot(All_metrics[0,:],All_metrics[1,:],marker='P' ,label = "train_acc")
    plt.plot(All_metrics[0,:],All_metrics[2,:],marker='P' ,label = "test_acc")
    plt.plot(All_metrics[0,:],All_metrics[3,:],marker='P' , label = "F1")
    plt.plot(All_metrics[0,:],All_metrics[4,:],marker='P' , label = "Recall")
    plt.plot(All_metrics[0,:],All_metrics[5,:],marker='P' , label = "Precision")
    plt.grid() 
    plt.xlabel('subjects')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracey with clssifier:{clf}')
      

    # Put a legend below current axis
    plt.legend(loc='upper right',fontsize = 'xx-small')
    #plt.show()
    plt.savefig(f'Accuracey/clssifier{clf}.png')


   


def conf_matrix(y_test, pred_test,sub,i,k):    
    
    # Creating a confusion matrix
    con_mat = confusion_matrix(y_test, pred_test)
    con_mat = pd.DataFrame(con_mat, range(2), range(2))
   
    #Ploting the confusion matrix
    plt.figure(figsize=(6,6))
    sns.set(font_scale=1.5) 
    g=sns.heatmap(con_mat, annot=True, annot_kws={"size": 16}, fmt='g', cmap='Blues', cbar=False)
    g.set_title(f'subj{sub}fold{i}nthtime{k}')
    plt.savefig(f'confmatSVR/subj{sub}fold{i}nthtime{k}')
    plt.close()
    
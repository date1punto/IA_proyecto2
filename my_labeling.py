__authors__ = ['1632398', '1633405', '1630320']
__group__ = 'DJ.12'

from utils_data import read_dataset, read_extended_dataset, crop_images, Plot3DCloud
from KNN import *
from Kmeans import *
import time
import matplotlib.pyplot as plt

def Kmean_statistics(KM, Kmax):
    KM.find_bestK(Kmax)
    millor_K=KM.K
    temps=[]
    iteracions=[]
    Distancia_intra_class=[]
    for x in range(2, Kmax+1):
        KM.K=x
        inici=time.time()
        KM.fit()
        fi=time.time()
        temps.append(fi-inici)
        iteracions.append(KM.num_iter)
        Distancia_intra_class.append(KM.withinClassDistance())
        
        
    #representacio de les dades en un nuvol de punt
    Plot3DCloud(KM)
    plt.show()
    
   
    #DISTANCIA INTRACLASS
    plt.plot(range(2,Kmax+1), Distancia_intra_class)
    plt.scatter(millor_K, Distancia_intra_class[millor_K-2],  c='red', marker='x')
    plt.ylabel("Distancia intra-class")
    plt.xlabel("Número centroides")
    plt.show()
    
    #TEMPS
    plt.plot(range(2,Kmax+1), temps)
    plt.scatter(millor_K, temps[millor_K-2],  c='red', marker='x')
    plt.ylabel("Temps")
    plt.xlabel("Número centroides")
    plt.show()
    
    #ITERACIONS
    plt.plot(range(2,Kmax+1), iteracions)
    plt.scatter(millor_K, iteracions[millor_K-2],  c='red', marker='x')
    plt.ylabel("Iteracions de la funció fit")
    plt.xlabel("Número centroides")
    plt.show()
    
if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    

    # You can start coding your functions here
    KM = KMeans(imgs[0],options={'km_init':'first'})
    
    Kmean_statistics(KM, 10)
    

#include <iostream>
#include<vector>
#include<math.h>

namespace machine_learning
{

/**
* Clusters your datapoints densety based, but single links are allowed. to take care if u realy want this...
*
* @param T first parameter is type to cluster
* @param X second is the distance type
* your cluster type needs overload + for center point...
* your cluster need overload -  as a distance function 
* calc the radius is not supported, because i guess it is not possible 4 me to make it generic, in case mass is != 1
* i am not possible to calc correct distance ..
* 
*/
template <class T,class X>
class DistanceConnectCluster{
    public:
        /**
            * @param points vector of points
            * @param r Distance
            */
        DistanceConnectCluster(const std::vector<T> &points,const X& r){
            distance = r;
            label.assign(points.size(),-1);
            
            int clusternr =0;
            for(size_t i=0;i<points.size();i++){
                if(label[i] == -1){
                    label[i]= clusternr;
                    clusternr++;
                }
                doPoint(points,i);

            }
            calcCenter(points,clusternr);
        }
        /**
            * label for each datapoint, that tells u to what cluster it belongs   
            **/
        std::vector<int> label;
        /**
            * is the point in your n space of the center of the cluster
            **/
        std::vector<T> center;
        /**
            * saves ur distance
            **/
        X distance;
    protected:
        
        void doPoint(const std::vector<T> &points,int i){
            for(size_t j=0;j<points.size();j++){
                 if(label[j] !=label[i]){
                    T d = points[i] - points[j];
                    if(d<distance){
                        label[j] =label[i];
                        doPoint(points,j);
                    }
                }
            }    
        }
        
        void calcCenter(const std::vector<T> &points,int cluster){
            center.assign(cluster,T());
            for(size_t i=0;i<points.size();i++){
                int cl = label[i];
                center[cl] = center[cl]+points[i];
            }
        }
 };	
	
	
}
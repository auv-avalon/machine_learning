

#include <iostream>
#include<vector>
#include<math.h>

#include"../src/DistanceConnectCluster.hpp"

/**
should possible also be possible with other data types like vectors 
*/
struct Point2f{
    /**
    * need + to befined if u what to have the center....
    **/
    Point2f operator+(const Point2f& p){
        Point2f ret = *this;
        ret.x += p.x*p.m;
        ret.y += p.y*p.m;
        ret.m += p.m;
        return ret;
    }
    /**
    * need - for the distance function !
    */
    Point2f operator-(const Point2f& p)const{
        Point2f ret = *this;
        ret.x -= p.x;
        ret.y -= p.y;
        return ret;
    }
    
    /**
    * has to know what the difference is
    */
    bool operator<(const float& p){
        if(sqrt(x*x+y*y)<p ){
            return true;
        }
        return false;
    }

    /**
    * only needed if the mass is != 1
    */
    void normalize(){
        x /=m;
        y /=m;
    }

    
    Point2f():x(0),y(0),m(0){};
          
    Point2f(float _x,float _y,float _m):x(_y),y(_y),m(_m){};
    Point2f(int _x,int _y, int _m):x(_y),y(_y),m(_m){};
    Point2f(int _x,int _y):x(_y),y(_y),m(1){};
        float x;
        float y;
        float m;
    
};

int main(){
    std::cout<<"hello\n";
    std::vector<Point2f> points= { 
        Point2f(0,1),
        Point2f(0,2),
        Point2f(0,4),
        Point2f(1,1),
        Point2f(2,2),
        Point2f(2,1),
        Point2f(10.0f,10.0f,0.1f),
        Point2f(12,10,0.1),
        Point2f(11,10,0.3),
        Point2f(11,11,100),
        Point2f(12.0f,12.0f,0.5f),
        Point2f(11,12,7)
        };

            
         machine_learning::DistanceConnectCluster<Point2f,float> c(points,3.0);
                
        std::cout<<c.label.size()<<" points\n";
        
        for(int i=0;i<c.center.size();i++){
            c.center[i].normalize(); // used to because of mass is != 1
            std::cout<<"center from cluster <<"<<i<<" x:"<< c.center[i].x<<" y:"<<c.center[i].y <<"\n";
        }

    return 1;
    
}


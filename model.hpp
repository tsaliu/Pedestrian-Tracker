#ifndef MODEL_H_INCLUDE
#define MODEL_H_INCLUDED

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <armadillo>

using namespace arma;

void model(float T, int sigv, mat &F, mat &Q, int px, int cx, int py, int cy, float &dx, float &dy){
    int order = 2;
    mat G(4,1);
    for(int i=0; i<(2*order);){
        F(i,i)=1;
        F(i,i+1)=T;
        F(i+1,i+1)=1;
        G(i,0)=pow(T,2)/2;
        G(i+1,0)=T;
//        F[i][i]=1;
//        F[i][i+1]=T;
//        F[i+1][i+1]=1;
//        G[i]=pow(T,2)/2;
//        G[i+1]=T;
        i=i+2;
    }

//    for(int k=0; k<(2*order)){
//        Q[k][k]=G[k]*G[k];
//        Q[k][k+1]=G[k]*G[k+1];
//        Q[k+1][k]=Q[k][k+1];
//        Q[k+1][k+1]=G[k+1]*G[k+1];
//        k=k+2;
//    }
    Q=G*pow(sigv,2)*G.t();

    dx=(cx-px)/T;
    dy=(cy-py)/T;


}
#endif

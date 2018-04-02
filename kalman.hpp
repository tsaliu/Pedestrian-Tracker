#ifndef KALMAN_H_INCLUDE
#define KALMAN_H_INCLUDED

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <armadillo>
#include "model.hpp"
using namespace std;
using namespace arma;

inline
void kalman(arma::mat &p, arma::mat &hxk1k1, int px, int cx, int py, int cy, float T, float R){

    int sigv=sqrt(25);
    arma::mat F(4,4);
    arma::mat Q(4,4);
    F.fill(0);
    Q.fill(0);
    float dx,dy;

    model(T,sigv,F,Q,px,cx,py,cy,dx,dy);

    arma::mat H(1,4);
    H(0,0)=1;
    H(0,1)=0;
    H(0,2)=1;
    H(0,3)=0;

    arma::mat w=randn(4,4);
    for(int i=0; i<w.n_rows; i++){
        for(int j=0; j<w.n_cols; j++){
            w(i,j)=w(i,j)*sqrt(Q(i,j));
        }
    }
    arma::mat v=randn(1,4);
    for(int i=0; i<v.n_cols; i++){
        v(0,i)=v(0,i)*sqrt(R);
    }
    arma::mat x00(4,4);
    x00=hxk1k1;
    x00(1,1)=dx;
    x00(3,3)=dy;
    arma::mat xx(4,4);
    xx=x00+w;

    arma::mat zk1(1,4);
    zk1=H*xx+v;
    arma::mat hxkk(4,4);
    hxkk=xx;
    arma::mat p00(4,4);
    p00=p;
    arma::mat pkk(4,4);
    pkk=p00;

    arma::mat hxk1k(4,4);
    hxk1k=F*hxkk;
    arma::mat hzk1k(1,4);
    hzk1k=H*hxk1k;
    arma::mat vk1(1,4);
    vk1=zk1-hzk1k;

    arma::mat pk1k(4,4);
    pk1k=F*pkk*F.t()+Q;
    float sk1;
    arma::mat sk11(1,1);
    sk11=R+H*pk1k*H.t();
    sk1=sk11(0,0);
    arma::mat wk1(4,1);
    wk1=pk1k*H.t()/sk1;

    arma::mat pk1k1(4,4);
    pk1k1=pk1k-wk1*sk1*wk1.t();
    p=pk1k1;

    hxk1k1=hxk1k+wk1*vk1;

}
#endif

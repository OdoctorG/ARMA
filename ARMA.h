#pragma once
#include <Arma/include/Eigen/Dense>


using Eigen::MatrixXd;
using Eigen::VectorXd;

#define DllExport __declspec( dllexport )

extern "C" {
	DllExport double* _ARparameters(int n, int p, double* X);
	DllExport double* _toARMAparameters(double* k, int n);
	DllExport double* _ARMAparameters(int p, int  n, double* x);
	DllExport double _ARforecast(double* x, int n1, double* thetaphi, int n2);
	DllExport double _ARMAforecast(double* x, int n1, double* thetaphi, int n2);
}

VectorXd ARparameters(int n, int p, VectorXd X);
VectorXd toARMAparameters(VectorXd k);
VectorXd ARMAparameters(int p, VectorXd x);
double ARforecast(VectorXd x, VectorXd thetaphi);
double ARMAforecast(VectorXd x, VectorXd thetaphi);

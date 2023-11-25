// ARMA.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include "ARMA.h"
#include <Arma/include/Eigen/Dense>
 
using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd ARparameters(int n, int p, VectorXd X) {
	// Vector X is of length of n . Int p is the number.
	
	MatrixXd U(n-p, p);
	for (int x = 1; x <= p; x++) {
		for (int y = 1; y <= n - p; y++) {
			U(y-1, x-1) = -X(p - x + y - 1);
		}
	}
	//std::cout << U << std::endl;
	VectorXd out = (U.transpose() * U).ldlt().solve(U.transpose() * X(Eigen::seq(p, n-1)));
	return out;
}

double* _ARparameters(int n, int p, double* X) {
	VectorXd v = Eigen::Map<VectorXd>(X,n);
	return ARparameters(n, p, v).data();
}

int oddeven(int n) {
	return 1 - 2 * (n % 2);
}

VectorXd toARMAparameters(VectorXd k) {
	int n = k.size();
	
	VectorXd thetaphi(n);
	double theta = 1;
	int len = 0;
	for (int i = 0; i < n-1; i++) {
		if ((k(i) != 0) && (abs( - k(i + 1) / k(i)) < abs(theta)) && abs(-k(i + 1) / k(i)) < 1.0) {
			theta += -k(i + 1) / k(i);
			len += 1;
		}
	}
	if (len > 0) { theta = theta / len; }
	
	thetaphi(0) = theta;
	for (int i = 1; i < n; i++) {
		thetaphi(i) = k(i) + oddeven(i + 1) * pow(theta, i);
		for (int j = 1; j < i; j++) {
			thetaphi(i) += thetaphi(j)* pow(theta, i - j)* oddeven(j + 1);
		}
	}
	return thetaphi(Eigen::seq(0, int(n / 2)));
	//return thetaphi;
}

double* _toARMAparameters(double* k, int n) {
	VectorXd v = Eigen::Map<VectorXd>(k,n);
	return toARMAparameters(v).data();
}


VectorXd ARMAparameters(int p, VectorXd x) {
	int n = x.size();
	VectorXd param = ARparameters(n, p * 2, x);
	return toARMAparameters(param);
}

double* _ARMAparameters(int p, int n, double* x) {
	VectorXd v = Eigen::Map<VectorXd>(x, n);
	return ARMAparameters(p,v).data();
}

double ARforecast(VectorXd x, VectorXd thetaphi) {
	VectorXd thetarev = thetaphi.reverse();
	int n = thetaphi.size();

	double pred = x(0);
	for (int i = 0; i < x.size(); i++) {
		int l = i - n + 1;
		if (l < 0) {
			l = 0;
		}
		VectorXd previous_data = x(Eigen::seq(l, i));
		int ll = n - i - 1;
		if (ll < 0) {
			ll = 0;
		}
		VectorXd thetas = thetarev(Eigen::seq(ll, n - 1));
		pred = -thetas.dot(previous_data);
	}
	return pred;
}

double _ARforecast(double* x, int n1, double* thetaphi, int n2) {
	VectorXd v1 = Eigen::Map<VectorXd>(x, n1);
	VectorXd v2 = Eigen::Map<VectorXd>(thetaphi, n2);
	return (ARforecast(v1, v2));
}

double ARMAforecast(VectorXd x, VectorXd thetaphi) {
	VectorXd thetarev = thetaphi.reverse();
	int n = thetaphi.size();
	
	double pred = x(0);
	for (int i = 0; i < x.size(); i++) {
		double residual = x(i) - pred;
		int l = i-n+2;
		if (l < 0) {
			l = 0;
		}
		VectorXd previous_data = x(Eigen::seq(l, i));
		int ll = n - i - 2;
		if (ll < 0) {
			ll = 0;
		}
		VectorXd thetas = thetarev(Eigen::seq(ll, n - 2));
		pred = -thetas.dot(previous_data) + thetaphi(0) * residual;
	}
	return pred;
}

double _ARMAforecast(double* x, int n1, double* thetaphi, int n2) {
	VectorXd v1 = Eigen::Map<VectorXd>(x, n1);
	VectorXd v2 = Eigen::Map<VectorXd>(thetaphi, n2);
	return ARMAforecast(v1, v2);
}


int main()
{
	VectorXd v(10); 
	v = VectorXd::LinSpaced(10,1,10);
	int n = 10;
	int p = 2;

	std::cout << "AR(p) forecast: " << ARforecast(v, ARparameters(n, p, v)) << std::endl;
	std::cout << "Arma(p-1,1) forecast: " << ARMAforecast(v, ARMAparameters(p,v)) << std::endl;
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

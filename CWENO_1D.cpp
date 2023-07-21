// CWENO_1D.cpp: 定义应用程序的入口点。
//

#include "CWENO_1D.h" //包含头文件
#include <iostream>
#include <cmath>
#include <matplotlibcpp.h>
#include <armadillo>
#include <string>
using namespace std;
using namespace arma;
namespace plt = matplotlibcpp;
//声明函数
void Rjx(vec& I1, vec& omega1, vec& omega2, vec& omega3, vec& Ruj, const vec& u, double h, int option);
void weight(const vec& u, int option, vec& omega1, vec& omega2, vec& omega3);
void I2_RK(mat& I2, vec& u, double h, double dt, int option, int value);
void derivf(vec& df, const vec& u, double h);
int main() {

	int N = 80;// 网格数
	double h = 2.0 / (N - 1);// 网格宽度
	double dt = 0.9 * 2.0 / 7 * h;//时间步长
	int tn = round(10 / dt);// 时间步数
	int origin;// 原点
	vec x = linspace<vec>(-1.0, 1.0, N);//生成0到1之间长度为N的等差数列
	vec x1 = x + h / 2.0;

	mat u2(N, tn, fill::zeros);//理论值
	vec t = linspace<vec>(0, tn * dt, tn + 1);
	//计算理论解
	for (int i = 0; i < tn; i++) {
		for (int j = 0; j < N; j++) {
			int option = i % 2;
			switch (option) {
			case 0:
				u2(j, i) = sin(datum::pi * (x(j) - t(i)));
				break;
			case 1:
				u2(j, i) = sin(datum::pi * (x1(j) - t(i)));
				break;
			}
		}
	}
	u2.save("u2.csv", csv_ascii);

	origin = 1; int option;

	//计算模拟值
	mat u(N, tn, fill::zeros);  // 存储模拟结果
	vec I0(N), Ruj(N); // 临时变量
	vec omega1(N), omega2(N), omega3(N);// 存储各种权重系数
	vec I1(N), I2(N);   // 存储临时变量
	// 初始条件
	for (int i = 0; i < N; i++)
	{
		u(i, 0) = sin(datum::pi * x(i));
	}
	// 主要计算循环
	for (int t = 1; t < tn; t++)
	{
		if (origin == 1)
		{
			option = (t + 1) % 2;
		}
		else
		{
			option = 3;
		}
		// 调用Rjx函数计算I1和Ruj
		Rjx(I1, omega1, omega2, omega3, Ruj, u.col(t - 1), h, option);
		// 调用I2_RK函数计算I2
		I2_RK(I2, Ruj, h, dt, option, 4);
		I0 = I1 + I2;
		u.col(t) = I0;
	}
	u.save("u.csv", csv_ascii);

	//cout << "u:" << endl << u << endl;
	//可视化输出

	for (int i = 0; i < tn; i++) {
		if (origin == 1)
			option = i % 2;
		else
			option = 3;
		//转为std::vec
		typedef std::vector<double> stdvec;
		stdvec ux = conv_to< stdvec >::from(u.col(i));
		stdvec x1x = conv_to< stdvec >::from(x1);
		stdvec xx = conv_to< stdvec >::from(x);
		
		switch (option)
		{

		case 1:
			plt::plot(x1x, ux);
			break;
		case 0:
			plt::plot(xx, ux);
			break;
		case 3:
			plt::plot(xx, ux);
			break;
		}

		//plt::title("u_t + u_x = 0, t = "+ char(i));
		//plt::grid(true);
		plt::xlim(-1, 1);
		plt::pause(0.002);
		//	//plt::show(false);

		//}
		return 0;
	}
}
void Rjx(vec& I1, vec& omega1, vec& omega2, vec& omega3, vec& Ruj, const vec& u, double h, int option)
{
	int n = u.n_elem;
	vec uj(n, fill::zeros), duj(n, fill::zeros), dduj(n, fill::zeros);
	vec Rduj(n, fill::zeros), Rdduj(n, fill::zeros);

	// Calculate omega1, omega2, omega3 using weight function
	weight(u, 1, omega1, omega2, omega3);

	// Calculate reconstructed u_k, u_k' and u_k''
	for (int i = 1; i < n - 1; i++)
	{
		uj(i) = u(i) - (u(i - 1) - 2 * u(i) + u(i + 1)) / 24.0;
		duj(i) = (u(i + 1) - u(i - 1)) / (2.0 * h);
		dduj(i) = (u(i + 1) - 2 * u(i) + u(i - 1)) / (h * h);
	}

	// Handle endpoints separately
	uj(0) = u(0) - (u(n - 2) - 2 * u(0) + u(1)) / 24.0;
	uj(n - 1) = uj(0);
	duj(0) = (u(1) - u(n - 2)) / (2.0 * h);
	duj(n - 1) = duj(0);
	dduj(0) = (u(1) - 2 * u(0) + u(n - 2)) / (h * h);
	dduj(n - 1) = dduj(0);
	////在e-13次方左右有误差
	//cout << "duj = " << endl << duj << endl;
	//cout << "dduj = " << endl << dduj << endl;
	//cout << "uj = " << endl << uj << endl;

	for (int i = 1; i < n - 1; i++)
	{
		Ruj(i) = omega1(i) * (uj(i - 1) + h * duj(i - 1) + 0.5 * h * h * dduj(i - 1)) + omega2(i) * uj(i) + omega3(i) * (uj(i + 1) - h * duj(i + 1) + 0.5 * h * h * dduj(i + 1));
		Rduj(i) = omega1(i) * (duj(i - 1) + h * dduj(i - 1)) + omega2(i) * duj(i) + omega3(i) * (duj(i + 1) - h * dduj(i + 1));
		Rdduj(i) = omega1(i) * dduj(i - 1) + omega2(i) * dduj(i) + omega3(i) * dduj(i + 1);
	}

	Ruj(0) = omega1(0) * (uj(n - 2) + h * duj(n - 2) + 0.5 * h * h * dduj(n - 2)) + omega2(0) * uj(0) + omega3(0) * (uj(1) - h * duj(1) + 0.5 * h * h * dduj(1));
	Ruj(n - 1) = Ruj(0);
	Rduj(0) = omega1(0) * (duj(n - 2) + h * dduj(n - 2)) + omega2(0) * duj(0) + omega3(0) * (duj(1) - h * dduj(1));
	Rduj(n - 1) = Rduj(0);
	Rdduj(0) = omega1(0) * dduj(n - 2) + omega2(0) * dduj(0) + omega3(0) * dduj(1);
	Rdduj(n - 1) = Rdduj(0);


	switch (option) {
	case 0:
		for (int i = 0; i < n - 1; i++) {
			I1(i) = (Ruj(i) + Ruj(i + 1)) / 2 - (Rduj(i + 1) - Rduj(i)) / 8 * h + (Rdduj(i) + Rdduj(i + 1)) / 48 * h * h;
		}
		I1(n - 1) = I1(0);
		break;

	case 1:
		for (int i = 1; i < n; i++) {
			I1(i) = (Ruj(i - 1) + Ruj(i)) / 2 - (Rduj(i) - Rduj(i - 1)) / 8 * h + (Rdduj(i - 1) + Rdduj(i)) / 48 * h * h;
		}
		I1(0) = I1(n - 1);
		break;

	case 3:
		for (int i = 0; i < n; i++) {
			I1(i) = Ruj(i) + h * h / 24 * Rdduj(i);
		}
		break;
	}
}
void weight(const vec& u, int option, vec& omega1, vec& omega2, vec& omega3) {
	int n = u.n_elem;
	vec ISj1(n), ISj2(n), ISj3(n), alpha1(n), alpha2(n), alpha3(n);
	omega1.resize(n); omega2.resize(n); omega3.resize(n);
	// Calculate ISj
	for (int i = 2; i <= n - 1; i++) {
		ISj1(i) = 13.0 / 12.0 * pow((u(i - 2) - 2 * u(i - 1) + u(i)), 2) + 1.0 / 4.0 * pow((u(i - 2) - 4 * u(i - 1) + 3 * u(i)), 2);
	}
	ISj1(1) = 13.0 / 12.0 * pow((u(n - 2) - 2 * u(0) + u(1)), 2) + 1.0 / 4.0 * pow((u(n - 2) - 4 * u(0) + 3 * u(1)), 2);
	ISj1(0) = 13.0 / 12.0 * pow((u(n - 3) - 2 * u(n - 2) + u(0)), 2) + 1.0 / 4.0 * pow((u(n - 3) - 4 * u(n - 2) + 3 * u(0)), 2);
	for (int i = 1; i <= n - 2; i++) {
		ISj2(i) = 13.0 / 12.0 * pow((u(i - 1) - 2 * u(i) + u(i + 1)), 2) + 1.0 / 4.0 * pow((u(i - 1) - u(i + 1)), 2);
	}
	ISj2(0) = 13.0 / 12.0 * pow((u(n - 2) - 2 * u(0) + u(1)), 2) + 1.0 / 4.0 * pow((u(n - 2) - u(1)), 2);
	ISj2(n - 1) = ISj2(0);
	for (int i = 0; i <= n - 3; i++) {
		ISj3(i) = 13.0 / 12.0 * pow((u(i) - 2 * u(i + 1) + u(i + 2)), 2) + 1.0 / 4.0 * pow((3 * u(i) - 4 * u(i + 1) + u(i + 2)), 2);
	}
	ISj3(n - 2) = 13.0 / 12.0 * pow((u(n - 2) - 2 * u(n - 1) + u(1)), 2) + 1.0 / 4.0 * pow((3 * u(n - 2) - 4 * u(n - 1) + u(1)), 2);

	ISj3(n - 1) = 13.0 / 12.0 * pow((u(n - 1) - 2 * u(1) + u(2)), 2) + 1.0 / 4.0 * pow((3 * u(n - 1) - 4 * u(1) + u(2)), 2);

	//// Calculate omega_j
	//cout << "ISj1 = " << endl << ISj1 << endl;
	//cout << "ISj2 = " << endl << ISj2 << endl;
	//cout << "ISj3 = " << endl << ISj3 << endl;
	vec C = zeros(3);
	double xi = 1e-6;
	int p = 2;
	for (int i = 0; i < n; i++) {
		if (option == 1) {
			C(0) = 3.0 / 16.0;
			C(1) = 5.0 / 8.0;
			C(2) = 3.0 / 16.0;
		}
		if (option == 2) {
			C(0) = 1.0 / 6.0;
			C(1) = 2.0 / 3.0;
			C(2) = 1.0 / 6.0;
		}
		double alpha1 = C(0) / pow(xi + ISj1(i), p);
		double alpha2 = C(1) / pow(xi + ISj2(i), p);
		double alpha3 = C(2) / pow(xi + ISj3(i), p);
		double talpha = alpha1 + alpha2 + alpha3;
		omega1(i) = alpha1 / talpha;
		omega2(i) = alpha2 / talpha;
		omega3(i) = alpha3 / talpha;
	}
}
void I2_RK(mat& I2, vec& u, double h, double dt, int option, int value) {

	u = u(span(0, u.n_elem - 1)); // 去除 u 中的冗余维度

	int n = u.n_elem;
	double lambda = dt / h; // Test 1, f(u)=u 
	vec un, uh, uf, I0;
	if (value == 2) {
		vec K1, K2;
		derivf(K1, u, h);
		derivf(K2, u + dt * K1, h);
		un = u;
		uh = u + dt * (3.0 / 8 * K1 + 1.0 / 8 * K2);
		uf = u + dt * (1.0 / 2 * K1 + 1.0 / 2 * K2);
	}
	else if (value == 4) {
		vec K1, K2, K3, K4;
		derivf(K1, u, h);
		derivf(K2, u + 0.5 * dt * K1, h);
		derivf(K3, u + 0.5 * dt * K2, h);
		derivf(K4, u + dt * K3, h);
		un = u;
		uh = u + dt * (5.0 / 24 * K1 + 1.0 / 6 * K2 + 1.0 / 6 * K3 - 1.0 / 24 * K4);
		uf = u + dt * (1.0 / 6 * K1 + 1.0 / 3 * K2 + 1.0 / 3 * K3 + 1.0 / 6 * K4);
	}
	else {
		throw std::invalid_argument("Invalid value for parameter 'value'.");
	}

	vec fun = un, fuh = uh, fuf = uf;

	switch (option) {
	case 0:
		I2.zeros(n);
		for (int i = 0; i < n - 1; i++) {
			I2(i) = lambda / 6 * (fun(i) + 4 * fuh(i) + fuf(i) - fun(i + 1) - 4 * fuh(i + 1) - fuf(i + 1));
		}
		I2(n - 1) = I2(0);
		break;
	case 1:
		I2.zeros(n);
		for (int i = 1; i < n; i++) {
			I2(i) = lambda / 6 * (fun(i - 1) + 4 * fuh(i - 1) + fuf(i - 1) - fun(i) - 4 * fuh(i) - fuf(i));
		}
		I2(0) = I2(n - 1);
		break;
	case 3:

		I2.zeros(n);
		for (int i = 0; i < n - 1; i++) {
			I2(i) = lambda / 6 * ((fun(i) + 4 * fuh(i) + fuf(i)) - (fun(i + 1) + 4 * fuh(i + 1) + fuf(i + 1)));
		}
		I2(n - 1) = I2(0);
		I0 = I2;
		for (int i = 1; i < n; i++) {
			I0(i) = (I2(i - 1) + I2(i)) / 2;
		}
		I0(0) = I0(n - 1);
		I2 = I0;
		break;
	default:
		cout << "Invalid option. Please enter 0, 1, or 3." << endl;
		break;
	}

}
void derivf(vec& df, const vec& u, double h) {
	int n = u.n_elem;
	vec f = u;
	vec fj = zeros<vec>(n);
	vec dfj = zeros<vec>(n);
	vec ddfj = zeros<vec>(n);
	for (int i = 1; i < n - 1; i++) {
		dfj(i) = (f(i + 1) - f(i - 1)) / (2 * h);
		ddfj(i) = (f(i + 1) - 2 * f(i) + f(i - 1)) / (h * h);
	}
	dfj(0) = (f(1) - f(n - 1)) / (2 * h);
	dfj(n - 1) = dfj(0);
	ddfj(0) = (f(1) - 2 * f(0) + f(n - 1)) / (h * h);
	ddfj(n - 1) = ddfj(0);

	vec omega1, omega2, omega3;
	weight(f, 2, omega1, omega2, omega3);

	// Calculate f derivative
	df = zeros<vec>(n);
	for (int i = 1; i < n - 1; i++) {
		df(i) = omega1(i) * (dfj(i - 1) + h * ddfj(i - 1)) + omega2(i) * dfj(i) + omega3(i) * (dfj(i + 1) - h * ddfj(i + 1));
	}
	df(0) = omega1(0) * (dfj(n - 1) + h * ddfj(n - 1)) + omega2(0) * dfj(0) + omega3(0) * (dfj(1) - h * ddfj(1));
	df(n - 1) = df(0);
	df *= -1.0;
}


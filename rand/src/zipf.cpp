#include <zipf.hh>
#include <cmath>

namespace zipf {

    double zeta(double theta, int n) {
        double sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += 1 / (pow(i, theta));
        }
        return sum;
    }

    int rand_zipf_r(unsigned int *seed, long n, double zetaN, double theta) {
        double alpha = 1 / (1 - theta);
        double eta = (1 - pow(2.0 / n, 1 - theta)) / (1 - zeta(theta, 2) / zetaN);
        double u = rand_r(seed) / (double) RAND_MAX;
        double uz = u * zetaN;
        if (uz < 1) return 1;
        if (uz < 1 + pow(0.5, theta)) return 2;
        return 1 + (int) (n * pow(eta * u - eta + 1, alpha));
    }

    int rand_zipf(long n, double zetaN, double theta) {
        double alpha = 1 / (1 - theta);
        double eta = (1 - pow(2.0 / n, 1 - theta)) / (1 - zeta(theta, 2) / zetaN);
        double u = rand() / (double) RAND_MAX;
        double uz = u * zetaN;
        if (uz < 1) return 1;
        if (uz < 1 + pow(0.5, theta)) return 2;
        return 1 + (int) (n * pow(eta * u - eta + 1, alpha));
    }
}
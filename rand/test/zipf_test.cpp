#include <zipf.hh>

int main() {

    const double zetan = zipf::zeta(0.99, 10000);

    while (zipf::rand_zipf(10000, zetan, 0.99) != 1)
        ;

    while (zipf::rand_zipf(10000, zetan, 0.99) == 1)
        ;

    return 0;
}
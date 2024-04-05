// Implemented by Logan Brenningmeyer

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>

using namespace cl::sycl;
using namespace std::chrono;

class totient;

unsigned int hcf(unsigned int x, unsigned int y)
{
  unsigned int t;

  while (y != 0) {
    t = x % y;
    x = y;
    y = t;
  }
  return x;
}


// relprime x y = hcf x y == 1

int relprime(unsigned int x, unsigned int y)
{
  return hcf(x, y) == 1;
}


// euler n = length (filter (relprime n) [1 .. n-1])

unsigned int euler(unsigned int n)
{
  unsigned int length, i;

  length = 0;
  for (i = 1; i < n; i++)
    if (relprime(n, i))
      length++;
  return length;
}


// sumTotient lower upper = sum (map euler [lower, lower+1 .. upper])

unsigned int sumTotient(unsigned int lower, unsigned int upper)
{
  unsigned int sum, i;

  sum = 0;
  for (i = lower; i <= upper; i++)
    sum = sum + euler(i);
  return sum;
}

int main() {

    for (int i = 0; i < 5; i++) {
        auto start = high_resolution_clock::now();

        unsigned int lower = 1;
        unsigned int upper = 15000;
        unsigned int size = upper - lower + 1;

        unsigned int sum = 0;

        {
            queue q;

            buffer<unsigned int> bufferSum{&sum, 1};

            try {
                q.submit([&](handler& cgh) {
                    auto sumReduction = reduction(bufferSum, cgh, plus<unsigned int>());

                    auto sum_kernel = [=](id<1> wID, auto& sum) {
                        unsigned int n = wID.get(0) + lower;
                        sum.combine(euler(n));
                    };

                    cgh.parallel_for<totient>(range<1>(size), sumReduction, sum_kernel);
                });
            } catch (sycl::exception const& e) {
                std::cout << "Caught synchronous SYCL exception:\n"
                        << e.what() << std::endl;
            }
        }

        std::cout << "Sum of totients: " << sum << std::endl;

        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);

        std::cout << "Time taken by function: "
            << duration.count() << " microseconds" << std::endl;

    }
    
    return 0;
}

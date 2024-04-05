// Implemented by Logan Brenningmeyer

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>

using namespace cl::sycl;
using namespace std::chrono;

class totient;

long hcf(long x, long y)
{
  long t;

  while (y != 0) {
    t = x % y;

    x = y;
    y = t;
  }
  return x;
}


// relprime x y = hcf x y == 1

int relprime(long x, long y)
{
  return hcf(x, y) == 1;
}


// euler n = length (filter (relprime n) [1 .. n-1])

long euler(long n)
{
  long length, i;

  length = 0;
  for (i = 1; i < n; i++)
    if (relprime(n, i))

      length++;
  return length;
}


// sumTotient lower upper = sum (map euler [lower, lower+1 .. upper])

long sumTotient(long lower, long upper)
{
  long sum, i;

  sum = 0;
  for (i = lower; i <= upper; i++)
    sum = sum + euler(i);
  return sum;
}


// ...

int main() {

    auto start = high_resolution_clock::now();

    try {
        long lower = 1;
        long upper = 100;
        long size = upper - lower + 1;

        std::vector<long> totients(size);

        {
            queue q(gpu_selector{});

            buffer<long, 1> totientBuffer(totients.data(), range<1>(size));

            q.submit([&](handler& cgh) {
                auto totientAccessor = totientBuffer.get_access<access::mode::write>(cgh);

                cgh.parallel_for<totient>(range<1>(size), [=](id<1> index) {
                    int i = index[0] + lower;

                    totientAccessor[index] = euler(i);
                });
            });
        }

        // Sum the totient values on the CPU
        long sum = 0;
        for (long i = 0; i < size; ++i) {
            printf("Totient %ld: %ld\n", i + lower, totients[i]);
            sum += totients[i];
        }

        std::cout << "Sum of totients: " << sum << std::endl;

        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);

        std::cout << "Time taken by function: "
             << duration.count() << " microseconds" << std::endl;
    } catch (cl::sycl::exception e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

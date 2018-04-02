#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <numeric>


int findmissing(std::vector<int> x, int number) {
    std::sort(x.begin(), x.end());
    auto pos = std::upper_bound(x.begin(), x.end(), number);

    if (*pos - number > 1)
        return number + 1;
    else {
        std::vector<int> diffs;
        std::adjacent_difference(pos, x.end(), std::back_inserter(diffs));
        auto pos2 = std::find_if(diffs.begin() + 1, diffs.end(), [](int x) { return x > 1; });
        return *(pos + (pos2 - diffs.begin() - 1)) + 1;
    }
}

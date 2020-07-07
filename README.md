# ORB-Features_Matches
Python bindings(and C++ code) for C++ implementation of ORB-Feature Extractor and ORB-Matcher, as implemented in <a href="https://github.com/raulmur/ORB_SLAM2"> ORB-SLAM2 </a>

## Steps for Python:

### To make orb_extractor
```
cd orb_extractor
```
Set corresponding numpy path for numpy in ```CMakeLists.txt line 13```
```
mkdir build
python3 gen2.py pyorb build headers.txt
cmake .
```
Go to ```build/pyorb_generated_include.h```. Change ```#include "src/ORB.hpp" ``` to ```#include "../src/ORB.hpp" ```

```make .```

Copy ```orb.so``` to parent folder.

### To make orb_matcher

Please follow the same steps for orb_extractor.
Change ```python3 gen2.py pyorb build headers.txt``` to ```python3 gen2.py pyorb_matcher build headers.txt```

Copy ```orb_matcher.so``` to parent folder.

### How to use:

```
import orb
import orb_matcher
OE = orb.orb_ORBextractor(2048,1.2,8,20,5)
kps,desc = OE.extract_orb_fts(im, None)

EM = orb_matcher.orb_matcher_ORBmatcher(float_val)
no_of_matches,matches = EM.find_matches(kp1,kp2,desc1,desc2,windowsize)
```

## To compile for C++
Example runner.cpp code.
```
cmake .
make
./Example
```

## Acknowledgements.

The C++ code has mostly been borrowed from ORB-SLAM2, in specific the <a href="https://github.com/raulmur/ORB_SLAM2/blob/master/src/ORBextractor.cc"> ORBextractor.cc </a> and <a href="https://github.com/raulmur/ORB_SLAM2/blob/master/src/ORBmatcher.cc.cc"> ORBmatcher.cc </a> with a few tiny changes (removed the concept of frames).
The code to create python bindings is from <a href = "https://www.learnopencv.com/how-to-convert-your-opencv-c-code-into-a-python-module/"> learnopencv.com</a>. The ```g++``` compilation code has been written equivalently in cmake. (to the best of my knowledge).

## Notes

1. This is a novice implementation, and can certainly be improved massively.
2. For some reason, the orb_matcher object has to be created everytime we wish to find matches. It gives wrong results otherwise. Need to investigate.
3. Will try to combine the two into a single module, which will be certainly easier to handle.

##### This is my first attempt attempt at combining C++ and Python implementations. Any suggestions are welcome.

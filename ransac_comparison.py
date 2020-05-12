# import numpy as np
#
# foo =np.random.choice(
#   ['pooh', 'rabbit', 'piglet', 'Christopher'],
#   1,
#   p=[0.9, 0.1, 0.0, 0.0]
# )
#
# print(foo)


# vm = np.loadtxt(sys.argv[1])
# mean_vm = vm.mean(0)

import numpy as np
# This is RANSAC Implementation
K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_portrait.txt")


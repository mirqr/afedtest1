import os
import sys

import config


if __name__ == "__main__":
    print(sys.argv)
    args = sys.argv[1:]
    for i in args:
        print(i)
    config_dict = config.get_config_dict()
    print(config_dict)

    #filename = "./foo/bar/baz.txt"
    #os.makedirs(os.path.dirname(filename), exist_ok=True)
    #with open(filename, "w") as f: pass
 

 
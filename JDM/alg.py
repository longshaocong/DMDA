from JDM.JDM import JDM
from JDM.Simsiam import CONTRA
from JDM.AR import AR

algorithms = {
    'JDM', 
    'CONTRA', 
    'AR'
}

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
                'algorithm not found: {}'.format(algorithm_name))
    return globals()[algorithm_name]
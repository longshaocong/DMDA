from JDM.JDM import JDM
from JDM.Simsiam import CONTRA
from JDM.AR import AR
from JDM.JDM_CON import JDM_con

algorithms = {
    'JDM', 
    'CONTRA', 
    'AR', 
    'JDM_con'
}

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
                'algorithm not found: {}'.format(algorithm_name))
    return globals()[algorithm_name]
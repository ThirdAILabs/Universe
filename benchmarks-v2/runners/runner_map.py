from .bolt_fc import BoltFullyConnectedRunner
from .dlrm import DLRMRunner
from .udt import UDTRunner
from .query_reformulation import QueryReformulationRunner

runner_map = {"bolt_fc": BoltFullyConnectedRunner, "dlrm": DLRMRunner, "udt": UDTRunner, "query_reformulation": QueryReformulationRunner}

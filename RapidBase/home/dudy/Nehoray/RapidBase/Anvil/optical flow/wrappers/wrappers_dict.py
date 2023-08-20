from .algorithm_wrappers import FlowNet1SimpleWrapper

WRAPPERS_DICT = {
    'FlowNet1Simple': FlowNet1SimpleWrapper,
}

""" types to support:
    Classic
    FlowNet1Simple
    FlowNet1Corr
    FlowNet2Simple
    FlowNet2Corr
    FastFlowNet
    DVDNet
    FastDVDNet
    IRR FlowNet
    IRR PWCNet
    RAFT
    Star Flow
    Ransac Flow
    SpyNet
    TVNet
    OF transformer"""


def get_optical_flow_algorithms_wrappers_dict():
    return WRAPPERS_DICT


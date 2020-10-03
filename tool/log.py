import sys
import os
import time
import logging

def GetRowNum():
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    return f.f_lineno

def GetPath():
    return os.getcwd()

def GetFileName():
    return os.path.basename(__file__) 

def GetFunName():
    print sys._getframe().f_code.co_name
    print sys._getframe().f_back.f_lineno 

def GetLogger(level):
    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter('[%(levelname)s %(asctime)s %(filename)s:%(lineno)d]  %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
	logging.getLogger().setLevel(level)
    return logger

if __name__ == '__main__':
    logger = GetLogger(logging.DEBUG)
    logger.warn('hello world')
